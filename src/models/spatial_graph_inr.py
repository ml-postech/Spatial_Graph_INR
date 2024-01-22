import torch
import torchmetrics as tm
import pytorch_lightning as pl

from argparse import ArgumentParser
from torch import nn
from torch.optim import lr_scheduler

from src.models.core import INR_MLP, parse_t_f



# Spatial graph INR model class
class SpatialGraphINR(pl.LightningModule):
    def __init__(self,
                 dataset_size: int, use_time: bool,
                 hyp_in_dim: int, hyp_hidden_dim: int, hyp_out_dim: int,
                 sph_in_dim: int, sph_hidden_dim: int, sph_out_dim: int,
                 euc_in_dim: int, euc_hidden_dim: int, euc_out_dim: int,
                 n_hyp_layers: int, n_sph_layers: int, n_euc_layers: int,
                 inr_out_dim: int, hidden_dim: int = 512, n_layers: int = 4,
                 input_concat: bool = True, latents: bool = False, latent_dim: int = 64, lambda_latent: float = 0.0001,
                 coord_mlp_geometric_init: bool = False, coord_mlp_sine: bool = True, coord_mlp_all_sine: bool = True, coord_mlp_beta: int = 0,
                 geometric_init: bool = False, sine: bool = True, all_sine: bool = True, beta: int = 0,
                 lr: float = 0.0005, lr_patience: int = 500,
                 skip: bool = True, bn: bool = False, dropout: float = 0.0, is_classifier: bool = False, **kwargs):
        super().__init__()
        
        self.dataset_size = dataset_size
        self.use_time = use_time
        
        self.hyp_in_dim = hyp_in_dim
        self.hyp_hidden_dim = hyp_hidden_dim
        self.hyp_out_dim = hyp_out_dim
        self.sph_in_dim = sph_in_dim
        self.sph_hidden_dim = sph_hidden_dim
        self.sph_out_dim = sph_out_dim
        self.euc_in_dim = euc_in_dim
        self.euc_hidden_dim = euc_hidden_dim
        self.euc_out_dim = euc_out_dim
        self.n_hyp_layers = n_hyp_layers
        self.n_sph_layers = n_sph_layers
        self.n_euc_layers = n_euc_layers
        
        self.inr_in_dim = 0
        self.inr_out_dim = inr_out_dim
        self.inr_hidden_dim = hidden_dim
        self.n_inr_layers = n_layers
        
        self.input_concat = input_concat
        self.use_latents = latents
        self.latent_dim = latent_dim
        self.lambda_latent: lambda_latent
        
        self.coord_mlp_geometric_init = coord_mlp_geometric_init
        self.coord_mlp_sine = coord_mlp_sine
        self.coord_mlp_all_sine = coord_mlp_all_sine
        self.coord_mlp_beta = coord_mlp_beta
        self.inr_geometric_init = geometric_init
        self.inr_sine = sine
        self.inr_all_sine = all_sine
        self.inr_beta = beta
        
        self.lr = lr
        self.lr_patience = lr_patience
        self.skip = skip
        self.bn = bn
        self.dropout = dropout
        self.is_classifier = is_classifier

        # Parallel processing is omitted
        # self.sync_dist = torch.cuda.device_count() > 1
        self.sync_dist = False
        
        # MLP models for treating coordinates separately before the INR model
        # If input_concat is set to False, these models are omitted.
        self.coordinate_mlp = None
        self.is_coordinate_exist = {'hyp': False, 'sph': False, 'euc': False}
        
        if hyp_in_dim > 0:
            self.is_coordinate_exist['hyp'] = True
        if sph_in_dim > 0:
            self.is_coordinate_exist['sph'] = True
        if euc_in_dim > 0:
            self.is_coordinate_exist['euc'] = True    
        
        if not input_concat:
            hyp_mlp = None
            sph_mlp = None
            euc_mlp = None
            
            if self.is_coordinate_exist['hyp']:
                hyp_mlp = INR_MLP(self.hyp_in_dim, self.hyp_out_dim, self.hyp_hidden_dim,
                                  self.n_hyp_layers, self.coord_mlp_geometric_init, self.coord_mlp_beta,
                                  self.coord_mlp_sine, self.coord_mlp_all_sine,
                                  self.skip, self.bn, self.dropout)
                
            if self.is_coordinate_exist['sph']:
                sph_mlp = INR_MLP(self.sph_in_dim, self.sph_out_dim, self.sph_hidden_dim,
                                  self.n_sph_layers, self.coord_mlp_geometric_init, self.coord_mlp_beta,
                                  self.coord_mlp_sine, self.coord_mlp_all_sine,
                                  self.skip, self.bn, self.dropout)
            
            if self.is_coordinate_exist['euc']:
                euc_mlp = INR_MLP(self.euc_in_dim, self.euc_out_dim, self.euc_hidden_dim,
                                  self.n_euc_layers, self.coord_mlp_geometric_init, self.coord_mlp_beta,
                                  self.coord_mlp_sine, self.coord_mlp_all_sine,
                                  self.skip, self.bn, self.dropout)
            
            self.coordinate_mlp = {'hyp': hyp_mlp, 'sph': sph_mlp, 'euc': euc_mlp}
            self.coordinate_mlp = nn.ModuleDict(self.coordinate_mlp)
            
            input_dim = self.hyp_out_dim + self.sph_out_dim + self.euc_out_dim
        else:
            input_dim = self.hyp_in_dim + self.sph_in_dim + self.euc_in_dim
        
        # Compute the input dimension for INR
        self.inr_in_dim = input_dim
        if self.use_time:
            self.inr_in_dim += 1
        if self.use_latents:
            self.inr_in_dim += latent_dim
            
        # INR part of the Model        
        self.inr = INR_MLP(self.inr_in_dim, self.inr_out_dim, self.inr_hidden_dim,
                           self.n_inr_layers, self.inr_geometric_init, self.inr_beta,
                           self.inr_sine, self.inr_all_sine,
                           self.skip, self.bn, self.dropout)
        
        # Latent codes
        if self.use_latents:
            self.latents = nn.Embedding(self.dataset_size, self.latent_dim)
        
        # Loss
        if self.is_classifier:
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()
            
        # Metrics
        self.r2_score = tm.R2Score(self.inr_out_dim)
        self.save_hyperparameters()
    
    # Model computation    
    def forward(self, emb_data):
        if not self.input_concat:            
            (hyp_emb, sph_emb, euc_emb) = torch.split(emb_data, [self.hyp_in_dim, self.sph_in_dim, self.euc_in_dim], dim = -1)
            
            hyp_mlp = self.coordinate_mlp['hyp']
            sph_mlp = self.coordinate_mlp['sph']
            euc_mlp = self.coordinate_mlp['euc']
            
            output_list = []
            
            if self.is_coordinate_exist['hyp']:
                hyp_output = hyp_mlp(hyp_emb)
                output_list.append(hyp_output)
            if self.is_coordinate_exist['sph']:
                sph_output = sph_mlp(sph_emb)
                output_list.append(sph_output)
            if self.is_coordinate_exist['euc']:
                euc_output = euc_mlp(euc_emb)
                output_list.append(euc_output)
            
            inr_input = torch.cat(output_list, -1)        
        else:
            inr_input = emb_data
        
        output = self.inr(inr_input)     
        return output
    
    def forward_with_preprocessing(self, data):
        points, indices = data
        if self.use_latents:
            latents = self.latents(indices)
            points = self.add_latent(points, latents)
        return self.forward(points)
    
    def add_latent(self, points, latents):
        n_points = points.shape[1]
        latents = latents.unsqueeze(1).repeat(1, n_points, 1)
        return torch.cat([latents, points], dim = -1)

    def latent_size_reg(self, indices):
        latent_loss = self.latents(indices).norm(dim = -1).mean()
        return latent_loss
    
    @staticmethod
    def gradient(inputs, outputs):
        d_points = torch.ones_like(outputs, requires_grad = False, device = outputs.device)
        points_grad = torch.autograd.grad(
            outputs = outputs,
            inputs = inputs,
            grad_outputs = d_points,
            create_graph = True,
            retain_graph = True,
            only_inputs = True,
        )[0][..., -3:]
        return points_grad

    def training_step(self, data):
        inputs, target, indices = data['inputs'], data['target'], data['index']

        # Add latent codes
        if self.use_latents:
            latents = self.latents(indices)
            inputs = self.add_latent(inputs, latents)

        inputs.requires_grad_()

        # Predict signal
        pred = self.forward(inputs)

        # Loss
        if self.is_classifier:
            pred = torch.permute(pred, (0, 2, 1))

        main_loss = self.loss_fn(pred, target)
        self.log('main_loss', main_loss, prog_bar = True, sync_dist = self.sync_dist)
        train_loss = main_loss

        if not self.is_classifier:
            self.r2_score(pred.view(-1, self.inr_out_dim), target.view(-1, self.inr_out_dim))
            self.log('r2_score', self.r2_score, prog_bar = True, on_epoch = True, on_step = False)

        # Latent size regularization
        if self.use_latents:
            latent_loss = self.latent_size_reg(indices)
            train_loss += self.lambda_latent * latent_loss
            self.log('latent_loss', latent_loss, prog_bar = True, sync_dist = self.sync_dist)

        self.log('train_loss', train_loss, sync_dist = self.sync_dist)

        return train_loss
    
    def validation_step(self, data, _):
        inputs, target, indices = data['inputs'], data['target'], data['index']
        
        if self.use_latents:
            latents = self.latents(indices)
            inputs = self.add_latent(inputs, latents)
            
        pred = self.forward(inputs)
        
        if self.is_classifier:
            pred = torch.permute(pred, (0, 2, 1))
            
        validation_loss = self.loss_fn(pred, target)
        self.log('validation_loss', validation_loss, prog_bar = True, sync_dist = self.sync_dist)
        return validation_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr = self.lr)

        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor = 0.5, patience = self.lr_patience, verbose = True)
        sch_dict = {'scheduler': scheduler, 'monitor': 'train_loss', 'frequency': 1}

        return {'optimizer': optimizer, 'lr_scheduler': sch_dict}
    
    @staticmethod
    def add_spatial_graph_inr_model_specific_args(parent_parser):
        parser = ArgumentParser(parents = [parent_parser], add_help = False)
        
        parser.add_argument('--input_concat', type = parse_t_f, default = True)
        parser.add_argument('--coord_mlp_geometric_init', type = parse_t_f, default = False)
        parser.add_argument('--coord_mlp_sine', type = parse_t_f, default = True)
        parser.add_argument('--coord_mlp_all_sine', type = parse_t_f, default = True)
        parser.add_argument('--coord_mlp_beta', type = int, default = 0)
        
        return parser
