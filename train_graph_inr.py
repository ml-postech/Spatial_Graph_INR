import os
import numpy as np
import torch
import torch.nn.functional as f
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from src.data.graph_dataset import GraphDataset, split_graphdataset
from src.models.graph_inr import GraphINR
from src.models.spatial_graph_inr import SpatialGraphINR
from src.plotting.figures import draw_pc
from src.utils.get_predictions import get_batched_predictions
from src.utils.load_emb_file import load_emb_info



if torch.cuda.is_available():
    accelerator = 'gpu'
    torch.set_float32_matmul_precision('high')
else:
    accelerator = 'cpu'

pl.seed_everything(1234)

parser = ArgumentParser()
parser.add_argument('--mode', default = 'fit', type = str)
parser.add_argument('--patience', default = 5000, type = int)
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--n_workers', default = 0, type = int)
parser.add_argument('--plot_3d', action = 'store_true')
parser.add_argument('--plot_heat', action = 'store_true')
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
parser = GraphINR.add_graph_inr_model_specific_args(parser)
parser = SpatialGraphINR.add_spatial_graph_inr_model_specific_args(parser)

args = parser.parse_args()

data_key = None
if 'us_elections' in args.dataset_dir:
    data_key = 'us_elections'
elif 'bunny' in args.dataset_dir:
    data_key = 'bunny'

if not args.mode in ['fit', 'pred']:
    raise ValueError('Mode should be one of the followings: fit, pred')

if not args.model in ['INR', 'GINR', 'Spatial_Graph_INR']:
    raise ValueError('Model should be one of the followings: INR, GINR, Spatial_Graph_INR')

if data_key == 'us_elections' and args.model == 'INR':
    raise ValueError('US-Election dataset does not have Cartesian coordinates to be used for INR')


# Load Data
dataset = GraphDataset(**vars(args))

if args.mode == 'pred':
    train_ratio = 0.8
    validation_ratio = 0.1
    test_ratio = 0.1
    
    train_dataset, validation_dataset, test_dataset = split_graphdataset(dataset, [train_ratio, validation_ratio, test_ratio])
    
    train = DataLoader(train_dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_workers)
    validation = DataLoader(validation_dataset, batch_size = args.batch_size, num_workers = args.n_workers)
    # test = DataLoader(test_dataset, batch_size = args.batch_size, num_workers = args.n_workers)
else:
    loader = DataLoader(dataset, batch_size = args.batch_size, shuffle = True, num_workers = args.n_workers)

if args.model == 'INR':
    input_dim = dataset.get_inputs(0).size(-1) + (1 if dataset.time else 0)
    output_dim = dataset.target_dim
    
    # Create a INR model based on the graph data loaded 
    model = GraphINR(input_dim, output_dim, len(dataset), **vars(args))
elif args.model == 'GINR':
    input_dim = dataset.n_fourier + (1 if dataset.time else 0)
    output_dim = dataset.target_dim
    
    # Create a GINR model based on the graph data loaded 
    model = GraphINR(input_dim, output_dim, len(dataset), **vars(args))
elif args.model == 'Spatial_Graph_INR':
    hyp_dim, hyp_copy, sph_dim, sph_copy, euc_dim, euc_copy = load_emb_info(args.emb_dir, args.emb_name)
    
    hyp_out_dim = 128
    sph_out_dim = 128
    euc_out_dim = 128
    
    output_dim = dataset.target_dim
    
    # Create a spatial graph INR model based on the graph embedding information loaded
    model = SpatialGraphINR(len(dataset), dataset.time,
                            ((hyp_dim + 1) * hyp_copy), 512, hyp_out_dim,
                            ((sph_dim + 1) * sph_copy), 512, sph_out_dim,
                            (euc_dim * euc_copy), 512, euc_out_dim,
                            4, 4, 4, output_dim, **vars(args))
    

# Training
checkpoint_cb = ModelCheckpoint(monitor = 'validation_loss' if args.mode == 'pred' else 'train_loss', mode = 'min', save_last = True, filename = 'best')
earlystopping_cb = EarlyStopping(monitor = 'validation_loss' if args.mode == 'pred' else 'train_loss', patience = args.patience)
lrmonitor_cb = LearningRateMonitor(logging_interval = 'step')
logger = WandbLogger(project = args.model, save_dir = 'lightning_logs')
logger.experiment.log({'CUDA_VISIBLE_DEVICES': os.environ.get('CUDA_VISIBLE_DEVICES', None)})

trainer = pl.Trainer.from_argparse_args(
    args,
    max_epochs = -1 if args.max_epochs is None else args.max_epochs,
    log_every_n_steps = 1,
    callbacks = [checkpoint_cb, earlystopping_cb, lrmonitor_cb],
    logger = logger,
    accelerator = accelerator,
    devices = [0]
    
    # ddp strategy is decided not to be used
    
    # devices = torch.cuda.device_count(),
    # strategy = 'ddp' if torch.cuda.device_count() > 1 else None
)

if args.mode == 'pred':
    trainer.fit(model, train, validation)
else:
    trainer.fit(model, loader)

model = model.load_from_checkpoint(checkpoint_cb.best_model_path)

try:
    points = dataset.ginr_npzs[0]['points']
except KeyError:
    points = np.load(os.path.join(dataset.dataset_dir, 'points.npy'))

if data_key == 'bunny':
    if args.mode == 'pred':
        inputs = test_dataset.get_data(0)['inputs']
        target = test_dataset.get_data(0)['target']
    
        _, pred = get_batched_predictions(model, inputs, 0)
        
        print(f'MSE Loss: {f.mse_loss(torch.Tensor(pred), target).item():.6f}')
        print(f'R2 Score: {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}')
    else:
        inputs = dataset.get_data(0)['inputs']
        target = dataset.get_data(0)['target']
        
        _, pred = get_batched_predictions(model, inputs, 0)
        
        print(f'MSE Loss: {f.mse_loss(torch.Tensor(pred), target).item():.6f}')
        print(f'R2 Score: {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}')
    
    inputs = dataset.get_inputs(0)
    _, pred = get_batched_predictions(model, inputs, 0)

    fig = draw_pc(points, pred[:, 0], colorscale = 'Reds')
    fig.show()
    logger.experiment.log({'Scatter': fig})
else:
    if args.mode == 'pred':
        inputs = test_dataset.get_data(0)['inputs']
        target = test_dataset.get_data(0)['target']
    
        _, pred = get_batched_predictions(model, inputs, 0)
        
        print(f'MSE Loss: {f.mse_loss(torch.Tensor(pred), target).item():.6f}')
        print(f'R2 Score: {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}')
    else:
        inputs = dataset.get_data(0)['inputs']
        target = dataset.get_data(0)['target']
        
        _, pred = get_batched_predictions(model, inputs, 0)
        
        print(f'MSE Loss: {f.mse_loss(torch.Tensor(pred), target).item():.6f}')
        print(f'R2 Score: {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}')
