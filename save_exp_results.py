import torch
import torch.nn.functional as f
import pytorch_lightning as pl

from argparse import ArgumentParser

from src.data.graph_dataset import GraphDataset, split_graphdataset
from src.models.graph_inr import GraphINR
from src.models.spatial_graph_inr import SpatialGraphINR
from src.utils.get_predictions import get_batched_predictions



if torch.cuda.is_available():
    accelerator = 'gpu'
    torch.set_float32_matmul_precision('high')
else:
    accelerator = 'cpu'


parser = ArgumentParser()
parser.add_argument('--mode', default = None, type = str)
parser.add_argument('--tag', default = None, type = str)
parser.add_argument('--data', default = None, type = str)
parser.add_argument('--embedding', default = None, type = str)
parser.add_argument('--cp_epochs', default = -1, type = int)
parser.add_argument('--coord_hidden_dim', default = 32, type = int)
parser.add_argument('--coord_out_dim', default = 16, type = int)
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
parser = GraphINR.add_graph_inr_model_specific_args(parser)
parser = SpatialGraphINR.add_spatial_graph_inr_model_specific_args(parser)

args = parser.parse_args()



if args.mode not in ['init', 'fit', 'pred', 'super_res']:
    print(f'Invalid mode argument for experiment with tag {args.tag}')
    raise ValueError('Invalid mode - possible options are: \'init\', \'fit\', \'pred\', and \'super_res\'')


if args.mode == 'init':
    print('Initializing .txt files for storing experimental results')
    
    train_result_list_file = open('train_result_list.txt', 'w')
    val_result_list_file = open('val_result_list.txt', 'w')
    test_result_list_file = open('test_result_list.txt', 'w')
    
    train_result_list_file.write('Tag         MSELoss         R^2\n')
    val_result_list_file.write('Tag         MSELoss         R^2\n')
    test_result_list_file.write('Tag         MSELoss         R^2\n')
    
    train_result_list_file.close()
    val_result_list_file.close()
    test_result_list_file.close()
else:
    print(f'Saving results for experiment with tag {args.tag}')
    
    if args.data not in ['us_election', 'bunny_res2', 'bunny_res2_down', 'bunny_v1']:
        raise ValueError('Invalid data - possible options are: \'us_election\', \'bunny_res2\', \'bunny_res2_down\', and \'bunny_v1\'')

    if args.model not in ['INR', 'GINR', 'Spatial_Graph_INR']:
        raise ValueError('Invalid model - possible options are: \'INR\', \'GINR\', and \'Spatial_Graph_INR\'')
    
    if args.mode == 'fit':        
        args.n_fourier = 100
        args.lr = 0.0001
        args.n_layers = 6
    elif args.mode == 'pred':
        train_ratio = 0.8
        validation_ratio = 0.1
        test_ratio = 0.1
        
        args.n_fourier = 100
        args.lr = 0.0001
        args.n_layers = 6
    else:        
        args.n_fourier = 7
        args.lr = 0.001
        args.n_layers = 8


    if args.data == 'us_election':
        args.dataset_dir = 'dataset/us_elections'
        args.n_nodes_in_sample = -1
    elif args.data == 'bunny_res2':
        args.dataset_dir = 'dataset/bunny_res2'
        args.n_nodes_in_sample = 5000
    elif args.data == 'bunny_res2_down':
        args.dataset_dir = 'dataset/bunny_res2/super_resolution/original'
        args.n_nodes_in_sample = 5000
    else:
        args.dataset_dir = 'dataset/bunny_v1'
        args.n_nodes_in_sample = 5000


    if args.model == 'Spatial_Graph_INR':
        if args.embedding is None:
            raise ValueError('Embedding information should be given for experiments with Spatial Graph INR models')
        
        emb_factors = args.embedding.split('_')
        
        if not len(emb_factors) == 6:
            raise ValueError('Invalid string for embedding information')
        
        hyp_dim, hyp_copy, sph_dim, sph_copy, euc_dim, euc_copy = int(emb_factors[0]), int(emb_factors[1]), int(emb_factors[2]), int(emb_factors[3]), int(emb_factors[4]), int(emb_factors[5])
        
        if not args.cp_epochs % 100 == 0:
            raise ValueError('Invalid cp_epochs value - the value should be multiples of 100')
        
        if args.data == 'us_election':
            args.emb_dir = f'spatial_embeddings/US-county-fb/{args.embedding}/emb_info'
        elif args.data == 'bunny_res2' or args.data == 'bunny_res2_down':
            args.emb_dir = f'spatial_embeddings/bunny_res2/{args.embedding}/emb_info'
        else:
            args.emb_dir = f'spatial_embeddings/bunny_v1/{args.embedding}/emb_info'
        
        args.emb_name = f'emb_{args.cp_epochs}'
    

    pl.seed_everything(1234)
    
    dataset = GraphDataset(**vars(args))
    
    if args.mode == 'pred':
        train_dataset, validation_dataset, test_dataset = split_graphdataset(dataset, [train_ratio, validation_ratio, test_ratio])
    
    if args.model == 'Spatial_Graph_INR':
        hyp_hidden_dim = args.coord_hidden_dim
        sph_hidden_dim = args.coord_hidden_dim
        euc_hidden_dim = args.coord_hidden_dim
        
        hyp_out_dim = args.coord_out_dim
        sph_out_dim = args.coord_out_dim
        euc_out_dim = args.coord_out_dim

        output_dim = dataset.target_dim

        model = SpatialGraphINR(len(dataset), dataset.time,
                                ((hyp_dim + 1) * hyp_copy), hyp_hidden_dim, hyp_out_dim,
                                ((sph_dim + 1) * sph_copy), sph_hidden_dim, sph_out_dim,
                                (euc_dim * euc_copy), euc_hidden_dim, euc_out_dim,
                                4, 4, 4, output_dim, **vars(args))
    else:
        input_dim = dataset.n_fourier + (1 if dataset.time else 0)
        output_dim = dataset.target_dim
        
        model = GraphINR(input_dim, output_dim, len(dataset), **vars(args))
    
    
    checkpoint_path = f'lightning_logs/{args.model}/{args.tag}/checkpoints/best.ckpt'
    model = model.load_from_checkpoint(checkpoint_path)
    
    if args.mode == 'pred':
        val_result_list_file = open('val_result_list.txt', 'a')
        test_result_list_file = open('test_result_list.txt', 'a')
        
        inputs = validation_dataset.get_data(0)['inputs']
        target = validation_dataset.get_data(0)['target']
        _, pred = get_batched_predictions(model, inputs, 0)
        val_result_list_file.write(f'{args.tag}    {f.mse_loss(torch.Tensor(pred), target).item():.6e}    {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}\n')
        
        inputs = test_dataset.get_data(0)['inputs']
        target = test_dataset.get_data(0)['target']
        _, pred = get_batched_predictions(model, inputs, 0)
        test_result_list_file.write(f'{args.tag}    {f.mse_loss(torch.Tensor(pred), target).item():.6e}    {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}\n')
    
        val_result_list_file.close()
        test_result_list_file.close()
    else:
        train_result_list_file = open('train_result_list.txt', 'a')
        
        inputs = dataset.get_data(0)['inputs']
        target = dataset.get_data(0)['target']
        _, pred = get_batched_predictions(model, inputs, 0)
        train_result_list_file.write(f'{args.tag}    {f.mse_loss(torch.Tensor(pred), target).item():.6e}    {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}\n')
    
        train_result_list_file.close()
        