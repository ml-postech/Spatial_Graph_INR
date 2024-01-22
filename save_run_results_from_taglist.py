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

train_ratio = 0.8
validation_ratio = 0.1
test_ratio = 0.1

spatial_graph_embeddings = ['4_2_4_2_4_1', '4_1_4_1_4_1', '2_2_2_2_4_1', '2_2_2_1_4_1', '2_1_2_1_4_1']
us_election_num_epochs = ['500', '1000', '2000', '3000', '5000']
bunny_num_epochs = ['1000', '2000', '3000', '4000', '5000']


parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
parser = GraphINR.add_graph_inr_model_specific_args(parser)
parser = SpatialGraphINR.add_spatial_graph_inr_model_specific_args(parser)

args = parser.parse_args()

args.n_fourier = 100


taglist_file = open('exp_wandb_taglist.txt', 'r')
eval_result_list_file = open('eval_result_list.txt', 'w')
test_result_list_file = open('test_result_list.txt', 'w')

eval_result_list_file.write('Tag          MSELoss    R^2\n')
test_result_list_file.write('Tag          MSELoss    R^2\n')


# Exp 2 - US_Election experiments
args.dataset_dir = 'dataset/us_elections'
args.n_nodes_in_sample = -1
i = -1

while(True):
    i += 1
    tag = taglist_file.readline()[:-1]
    
    if tag[0] == '-':
        break
    elif tag == 'None':
        continue
    
    pl.seed_everything(1234)
    
    if i < 25:
        args.model = 'Spatial_Graph_INR'
        emb = spatial_graph_embeddings[i // 5]
        args.emb_dir = f'spatial_embeddings/US-county-fb/{emb}/emb_info'
        args.emb_name = f'emb_{us_election_num_epochs[i % 5]}'

        dataset = GraphDataset(**vars(args))
        train_dataset, validation_dataset, test_dataset = split_graphdataset(dataset, [train_ratio, validation_ratio, test_ratio])
        
        hyp_dim, hyp_copy, sph_dim, sph_copy, euc_dim, euc_copy = int(emb[0]), int(emb[2]), int(emb[4]), int(emb[6]), int(emb[8]), int(emb[10]), 
        
        hyp_out_dim = 128
        sph_out_dim = 128
        euc_out_dim = 128

        output_dim = dataset.target_dim

        model = SpatialGraphINR(len(dataset), dataset.time,
                                ((hyp_dim + 1) * hyp_copy), 512, hyp_out_dim,
                                ((sph_dim + 1) * sph_copy), 512, sph_out_dim,
                                (euc_dim * euc_copy), 512, euc_out_dim,
                                4, 4, 4, output_dim, **vars(args))
    elif i == 25:
        args.model = 'GINR'
    
        dataset = GraphDataset(**vars(args))
        train_dataset, validation_dataset, test_dataset = split_graphdataset(dataset, [train_ratio, validation_ratio, test_ratio])
        
        input_dim = dataset.n_fourier + (1 if dataset.time else 0)
        output_dim = dataset.target_dim
        
        model = GraphINR(input_dim, output_dim, len(dataset), **vars(args))
    else:
        break
    
    checkpoint_path = f'lightning_logs/{args.model}/{tag}/checkpoints/best.ckpt'
    model = model.load_from_checkpoint(checkpoint_path)
    
    inputs = validation_dataset.get_data(0)['inputs']
    target = validation_dataset.get_data(0)['target']
    _, pred = get_batched_predictions(model, inputs, 0)
    eval_result_list_file.write(f'{tag}     {f.mse_loss(torch.Tensor(pred), target).item():.6f}   {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}\n')
    
    inputs = test_dataset.get_data(0)['inputs']
    target = test_dataset.get_data(0)['target']
    _, pred = get_batched_predictions(model, inputs, 0)
    test_result_list_file.write(f'{tag}     {f.mse_loss(torch.Tensor(pred), target).item():.6f}   {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), target.view(-1, model.inr_out_dim)):.6f}\n')


taglist_file.close()
eval_result_list_file.close()
test_result_list_file.close()



args.dataset_dir = 'dataset/bunny_res2'
args.n_nodes_in_sample = 5000


'''
    elif i == 26:
        args.dataset_dir = 'dataset/us_elections'
        args.model = 'INR'
        
        dataset = GraphDataset(**vars(args))
        train_dataset, validation_dataset, test_dataset = split_graphdataset(dataset, [train_ratio, validation_ratio, test_ratio])
        
        input_dim = dataset.get_inputs(0).size(-1) + (1 if dataset.time else 0)
        output_dim = dataset.target_dim
        
        model = GraphINR(input_dim, output_dim, len(dataset), **vars(args))
'''

pl.seed_everything(1234)



pl.seed_everything(1234)