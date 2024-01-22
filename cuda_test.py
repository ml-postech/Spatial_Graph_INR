import os
import torch
import time

import pymesh
from src.utils.data_generation import (get_fourier, get_output_dir, load_mesh, mesh_to_graph, normalized_laplacian)
import numpy as np


mesh = load_mesh(f'./data_generation/bunny/reconstruction/bun_zipper_res2.ply')
mesh = pymesh.subdivide(mesh, order = 1)

points, edges, adj = mesh_to_graph(mesh)

print(points.shape[0])
print(edges.shape[0])

'''
device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')

#print(os.environ['CUDA_VISIBLE_DEVICES'])
print('Device: ', device)
print('Current cuda device: ', torch.cuda.current_device())
print('Count of using GPUs: ', torch.cuda.device_count())


data1 = np.load('dataset/bunny_res2/super_resolution/original/npz_files/data_res2_original.npz')
data2 = np.load('dataset/bunny_res2/npz_files/data_res2.npz')

print(data1['target'])
print(data2['target'];
'''