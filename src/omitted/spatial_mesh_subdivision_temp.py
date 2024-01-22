# Perform mesh subdivision with simple method in PyMesh package, with node features of spatial graph embeddings

import matplotlib.pyplot as plt
import numpy as np
import pymesh
import pytorch_lightning as pl
import torch

from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import r2_score
from tqdm import tqdm

from src.data.graph_dataset import GraphDataset
from src.models.spatial_graph_inr import SpatialGraphINR
from src.plotting.figures import draw_mesh, draw_pc
from src.utils.data_generation import load_mesh, mesh_to_graph
from src.utils.get_predictions import get_batched_predictions
from src.utils.load_emb_file import load_emb_info
from src.utils.polate_spatial_emb import get_spatial_emb_midpoint



# Read arguments
parser = ArgumentParser()
parser.add_argument('--checkpoint', default = '', type = str)
parser.add_argument('--mesh', default = 'data_generation/bunny/reconstruction/bun_zipper_res2.ply', type = str)
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
parser = SpatialGraphINR.add_spatial_graph_inr_model_specific_args(parser)

args = parser.parse_args()

args.model = 'Spatial_Graph_INR'
args.emb_dir = 'spatial_embeddings/bunny_res2/4_1_4_1_4_1'
args.emb_name = 'bunny_res2_4_1_4_1_4_1'
args.dataset_dir = 'dataset/bunny_res2'

# Data
dataset = GraphDataset(**vars(args))
mesh_train = load_mesh(args.mesh)
u_train = dataset.get_inputs(0).numpy()
y_train = dataset.get_target(0).numpy()

mesh_load = load_mesh(args.mesh)
original_graph = mesh_to_graph(mesh_load)
orig_nodes, orig_edges, _ = original_graph

print('Original mesh')
print('Number of nodes: ', len(orig_nodes))
print('Number of edges: ', len(orig_edges))
print('Number of faces: ', len(mesh_load.faces))

subdivided_mesh = pymesh.subdivide(mesh_load, order = 1)
subdivided_graph = mesh_to_graph(subdivided_mesh)
sub_nodes, sub_edges, _ = subdivided_graph

# Subdivided mesh obtained using simple method included in PyMesh
print('Subdivided mesh created using simple method')
print('Number of nodes: ', len(sub_nodes))
print('Number of edges: ', len(sub_edges))
print('Number of faces: ', len(subdivided_mesh.faces))

# Add new nodes and their corresponding embeddings in the subdivided mesh
# The order of new nodes are identical to that of subdivided mesh created with PyMesh
if args.model == 'dSpatial_Graph_INR':
    # The following is used for spatial graph INR - input is concatenated with the order of hyp, sph, and euc
    hyp_dim, hyp_copy, sph_dim, sph_copy, euc_dim, euc_copy = load_emb_info(args.emb_dir, args.emb_name)
    assert ((hyp_dim + 1) * hyp_copy) + ((sph_dim + 1) * sph_copy) + (euc_dim * euc_copy) == np.shape(u_train)[1], 'Embedding dimension does not match with given embedding information'
    emb_info = [hyp_dim, hyp_copy, sph_dim, sph_copy, euc_dim, euc_copy]

    new_nodes = np.empty((0, np.shape(u_train)[1]))
    
    for i in tqdm(range(len(mesh_load.faces)), total = len(mesh_load.faces), desc = 'Subdividing mesh with spatial graph embedding'):
        n1_idx = mesh_load.faces[i, 0]
        n2_idx = mesh_load.faces[i, 1]
        n3_idx = mesh_load.faces[i, 2]
        
        n1 = torch.Tensor(u_train[n1_idx])
        n2 = torch.Tensor(u_train[n2_idx])
        n3 = torch.Tensor(u_train[n3_idx])
        
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_spatial_emb_midpoint(n1, n2, emb_info, norm = 1.0), dim = 0).numpy(), axis = 0)
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_spatial_emb_midpoint(n2, n3, emb_info, norm = 1.0), dim = 0).numpy(), axis = 0)
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_spatial_emb_midpoint(n3, n1, emb_info, norm = 1.0), dim = 0).numpy(), axis = 0)
        
    old_nodes = u_train
elif args.model == 'Spatial_Graph_INR':
    # The following is used for classic INR - input is 2D or 3D coordinates
    new_nodes = np.empty((0, np.shape(mesh_load.nodes)[1]))
    
    for i in tqdm(range(len(mesh_load.faces)), total = len(mesh_load.faces), desc = 'Subdividing mesh with 3D coordinates'):
        n1_idx = mesh_load.faces[i, 0]
        n2_idx = mesh_load.faces[i, 1]
        n3_idx = mesh_load.faces[i, 2]
        
        n1 = mesh_load.nodes[n1_idx]
        n2 = mesh_load.nodes[n2_idx]
        n3 = mesh_load.nodes[n3_idx]
        
        new_nodes = np.append(new_nodes, np.expand_dims(np.mean([n1, n2], axis = 0), 0), axis = 0)
        new_nodes = np.append(new_nodes, np.expand_dims(np.mean([n2, n3], axis = 0), 0), axis = 0)
        new_nodes = np.append(new_nodes, np.expand_dims(np.mean([n3, n1], axis = 0), 0), axis = 0)
        
    old_nodes = orig_nodes
    
print(mesh_load.nodes)
print(type(mesh_load.nodes))

# Removing duplicate nodes
new_nodes, idx = np.unique(np.array(new_nodes), axis = 0, return_index = True)
new_nodes = new_nodes[np.argsort(idx)]
subdivided_mesh_nodes = np.concatenate([old_nodes, new_nodes], axis = 0)

print('Mesh subdivision with simple method and degree = 1 for spatial graph embedding has been completed')
