import torch
import torchmetrics as tm
import pytorch_lightning as pl

from argparse import ArgumentParser
from torch import nn
from torch.optim import lr_scheduler

from src.models.core import INR_MLP, parse_t_f



# GINR model class
class GraphINR(pl.LightningModule):
    '''
    Arguments:
        input_dim: int, size of the inputs
        output_dim: int, size of the ouputs
        dataset_size: int, number of samples in the dataset (for autodecoder latents)
        hidden_dim: int = 512, number of neurons in hidden layers
        n_layers: int = 4, number of layers (total, including first and last)
        lr: float = 0.0005, learning rate
        lr_patience: int = 500, learning rate patience (in number of epochs)
        latents: bool = False, make the model an autodecoder with learnable latents
        latent_dim: int = 256, size of the latents
        lambda_latent: float = 0.0001, regularization factor for the latents
        geometric_init: bool = False, initialize weights so that output is spherical
        beta: int = 0, if positive, use SoftPlus(beta) instead of ReLU activations
        sine: bool = False, use SIREN activation in the first layer
        all_sine: bool = False, use SIREN activations in all other layers
        skip: bool = True, add a skip connection to the middle layer
        bn: bool = False, use batch normalization
        dropout: float = 0.0, dropout rate
        classifier: bool = False, use CrossEntropyLoss as loss
    '''

    def __init__(self,
                 input_dim: int, output_dim: int,
                 dataset_size: int, hidden_dim: int = 512, n_layers: int = 4,
                 lr: float = 0.0005, lr_patience: int = 500,
                 latents: bool = False, latent_dim: int = 256, lambda_latent: float = 0.0001,
                 geometric_init: bool = False, beta: int = 0, sine: bool = False, all_sine: bool = False,
                 skip: bool = True, bn: bool = False, dropout: float = 0.0, is_classifier: bool = False, **kwargs):
        super().__init__()

        self.input_dim = input_dim
        self.inr_out_dim = output_dim
        self.dataset_size = dataset_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.lr = lr
        self.lr_patience = lr_patience
        self.use_latents = latents
        self.latent_dim = latent_dim
        self.lambda_latent = lambda_latent
        self.geometric_init = geometric_init
        self.beta = beta
        self.sine = sine
        self.all_sine = all_sine
        self.skip = skip
        self.bn = bn
        self.dropout = dropout
        self.is_classifier = is_classifier

        # Parallel processing is omitted
        # self.sync_dist = torch.cuda.device_count() > 1
        self.sync_dist = False

        # Compute true input dimension
        input_dim_true = self.input_dim
        if latents:
            input_dim_true += self.latent_dim

        # Modules
        self.model = INR_MLP(input_dim_true, self.inr_out_dim, self.hidden_dim,
                             self.n_layers, self.geometric_init, self.beta,
                             self.sine, self.all_sine,
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

    def forward(self, points):
        return self.model(points)

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
    def add_graph_inr_model_specific_args(parent_parser):
        parser = ArgumentParser(parents = [parent_parser], add_help = False)
        
        parser.add_argument('--hidden_dim', type=int, default = 512)
        parser.add_argument('--n_layers', type=int, default = 4)
        parser.add_argument('--lr', type = float, default = 0.0005)
        parser.add_argument('--lr_patience', type = int, default = 1000)
        parser.add_argument('--latents', action = 'store_true')
        parser.add_argument('--latent_dim', type = int, default = 256)
        parser.add_argument('--lambda_latent', type = float, default = 0.0001)
        parser.add_argument('--geometric_init', type = parse_t_f, default = False)
        parser.add_argument('--beta', type = int, default = 0)
        parser.add_argument('--sine', type = parse_t_f, default = False)
        parser.add_argument('--all_sine', type = parse_t_f, default = False)
        parser.add_argument('--skip', type = parse_t_f, default = True)
        parser.add_argument('--bn', type = parse_t_f, default = False)
        parser.add_argument('--dropout', type = float, default = 0.0)

        return parser
