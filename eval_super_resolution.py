'''
Evaluation script for the super-resolution experiment with the Stanford bunny.
Loads the training mesh from the data_generation folder and performs mesh
subdivision using PyMesh.
Automatically aligns the eigenvectors using the KL divergence of the histograms.

Arguments:
    - checkpoint: path to a Pytorch Lightning checkpoint file

Note: requires the --dataset_dir flag to be specified as well.
'''

import os

import matplotlib.pyplot as plt
import numpy as np
import pymesh
import pytorch_lightning as pl
import torch
import torch.nn.functional as f

from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import r2_score

from src.data.graph_dataset import GraphDataset
from src.models.graph_inr import GraphINR
from src.models.spatial_graph_inr import SpatialGraphINR
from src.plotting.figures import draw_mesh, draw_pc
from src.utils.data_generation import get_fourier, load_mesh, mesh_to_graph
from src.utils.eigenvectors import align_eigenvectors_kl
from src.utils.get_predictions import get_batched_predictions
from src.utils.spatial_mesh_subdivision import spatial_mesh_simple_subdivision, euc_mesh_simple_subdivision



parser = ArgumentParser()
parser.add_argument('--tag', default = None, type = str)
parser.add_argument('--checkpoint', default = '', type = str)
# bunny_res2 dataset will be used
parser.add_argument('--mode', default = 'original', type = str)
parser.add_argument('--mesh', default = 'data_generation/bunny/reconstruction/bun_zipper_res2.ply', type = str)
parser = pl.Trainer.add_argparse_args(parser)
parser = GraphDataset.add_dataset_specific_args(parser)
parser = GraphINR.add_graph_inr_model_specific_args(parser)
parser = SpatialGraphINR.add_spatial_graph_inr_model_specific_args(parser)

args = parser.parse_args()

if args.mode not in ['original', 'upsampling']:
    raise ValueError('Mode should be one of the followings: original and upsampling')

if args.mode == 'original':
    args.dataset_dir = 'dataset/bunny_res2'
else:
    args.dataset_dir = 'dataset/bunny_res2/super_resolution/original'

if args.tag is not None:
    args.checkpoint = f'lightning_logs/{args.model}/{args.tag}/checkpoints/best.ckpt'
    save_dir = f'super_resolution_results/{args.tag}/'
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
else:
    save_dir = None

# Load original mesh
dataset = GraphDataset(**vars(args))
mesh_train = load_mesh(args.mesh)
u_train = dataset.get_inputs(0).numpy()
y_train = dataset.get_target(0).numpy()

# Plot 1 - Plot the original signal
print('Plotting original signal')
rot = R.from_euler('xyz', [90, 00, 145], degrees = True).as_matrix()
fig = draw_mesh(mesh_train, intensity = y_train[:, 0], colorscale = 'Reds', rot = rot)
fig.update_layout(scene_camera = dict(eye = dict(x = 1.1, y = 1.1, z = 0.2)))
fig.write_html(f'{save_dir}bunny_signal_original_true.html')
# fig.show()

# Load model from checkpoint
if args.model == 'GINR' or args.model == 'INR':
    model = GraphINR.load_from_checkpoint(args.checkpoint)
elif args.model == 'Spatial_Graph_INR':
    model = SpatialGraphINR.load_from_checkpoint(args.checkpoint)
else:  
    raise ValueError('Inappropriate model type.')

# Plot 2 - Plot fitting results on the original mesh
print('Plotting fitting results on the original mesh')
inputs = torch.from_numpy(u_train).float()
_, pred = get_batched_predictions(model, inputs, 0)
fig = draw_mesh(mesh_train, intensity = pred, colorscale = 'Reds', rot = rot)
fig.update_layout(scene_camera = dict(eye=dict(x = 1.1, y = 1.1, z = 0.2)))
fig.write_html(f'{save_dir}bunny_signal_original_fit.html')
# fig.show()

# Obtain test data (subdivided mesh)
print('Obtaining subdivided mesh using simple method and order = 1')
mesh_test = pymesh.subdivide(mesh_train, order = 1)

if args.model == 'GINR':
    # Align eigenvectors to training ones
    _, _, adj_test = mesh_to_graph(mesh_test)
    u_test = get_fourier(adj_test, k = args.n_fourier)
    u_test = align_eigenvectors_kl(u_train, u_test)
elif args.model == 'INR':
    # Obtain Euclidean coordinates for new nodes using midpoint computations
    u_test = euc_mesh_simple_subdivision(args.mesh, u_train)
elif args.model == 'Spatial_Graph_INR':
    # Obtain embeddings for new nodes using midpoint computations
    u_test = spatial_mesh_simple_subdivision(args.mesh, u_train, args.emb_dir, args.emb_name)

# Predict signal
inputs = torch.from_numpy(u_test).float()
_, pred = get_batched_predictions(model, inputs, 0)

# Plot 3 - Plot test signal on the subdivided mesh
print('Plotting predicted signal on the super-resolved mesh')
fig = draw_mesh(
    mesh_test,
    intensity = pred,
    colorscale = 'Reds',
    rot = rot,
    cmin = y_train.min(),
    cmax = y_train.max(),
)
fig.update_layout(scene_camera = dict(eye = dict(x = 1.1, y = 1.1, z = 0.2)))
fig.write_html(f'{save_dir}bunny_signal_super-resolved_pred.html')
# fig.show()
    

# Plot 4 & 5 - Plot zoomed-in point clouds (take screenshots here!)
print('Plotting zoomed-in point clouds for the prediction on the original mesh')
zoom = 0.6  # Lower is more zoomed
inputs = torch.from_numpy(u_train).float()
_, pred = get_batched_predictions(model, inputs, 0)
fig = draw_mesh(mesh_train, rot = rot, color = 'black')
fig.update_layout(scene_camera = dict(eye = dict(x = zoom, y = zoom, z = 0.2)))
pc_trace = draw_pc(
    mesh_train.vertices * 1.001,
    color = pred[:, 0],
    colorscale = 'Reds',
    rot = rot,
    marker_size = 1.5,
).data[0]
fig.add_trace(pc_trace)
fig.write_html(f'{save_dir}bunny_point_cloud_original.html')
# fig.show()

print('Plotting zoomed-in point clouds for the prediction on the super-resolved mesh')
inputs = torch.from_numpy(u_test).float()
_, pred = get_batched_predictions(model, inputs, 0)
fig = draw_mesh(mesh_test, rot = rot, color = 'black')
fig.update_layout(scene_camera = dict(eye = dict(x = zoom, y = zoom, z = 0.2)))
pc_trace = draw_pc(
    mesh_test.vertices * 1.001,
    color = pred[:, 0],
    colorscale = 'Reds',
    rot = rot,
    marker_size = 1.5,
).data[0]
fig.add_trace(pc_trace)
fig.write_html(f'{save_dir}bunny_point_cloud_super-resolved.html')
# fig.show()


print('Computing statistics for the original mesh')

# Compute squared error per node
mse = (y_train - pred[:u_train.shape[0]]) ** 2

# Plot 6 - Plot error per node - original mesh only
print('Plotting mse error for the original mesh')
fig = draw_mesh(mesh_train, intensity = mse, colorscale = 'Reds', rot = rot)
fig.update_layout(scene_camera = dict(eye = dict(x = 1.1, y = 1.1, z = 0.2)))
fig.write_html(f'{save_dir}bunny_mse_loss_original.html')
# fig.show()

print(f'MSE Loss: {f.mse_loss(torch.Tensor(pred[:u_train.shape[0]]), torch.Tensor(y_train)).item()}')

# Compute r2
score = r2_score(y_train, pred[:u_train.shape[0]])
print(f'R2 score: {score}')

# Compute r2 without 90th percentile outliers
mask = mse < np.percentile(mse, 90)
r2_score_adjusted = r2_score(y_train[mask], pred[:u_train.shape[0]][mask])
print(f'R2 score adjusted (90th percentile): {r2_score_adjusted}')

# Compute r2 without 95th percentile outliers
mask = mse < np.percentile(mse, 95)
r2_score_adjusted = r2_score(y_train[mask], pred[:u_train.shape[0]][mask])
print(f'R2 score adjusted (95th percentile): {r2_score_adjusted}')

# Plot distribution of squared error
plt.figure(figsize = (2.2, 2.2))
plt.hist(mse, bins = 10, density = True)
plt.yscale('log')
plt.xlabel('Squared error')
plt.ylabel('Density')
plt.tight_layout()
plt.savefig(f'{save_dir}bunny_error_density_original.pdf', bbox_inches = 'tight')

# Upsampling tasks have true signal data for the super-resolved mesh
# Therefore, analysis on the prediction results on the super-resolved mesh is possible
if args.mode == 'upsampling':
    args.dataset_dir = 'dataset/bunny_res2/super_resolution/super_resolved'
    dataset = GraphDataset(**vars(args))
    y_test = dataset.get_target(0).numpy()
    
    # Plot 7 - Plot true signal on the subdivided mesh
    print('Plotting true signal on the super-resolved mesh')
    fig = draw_mesh(
        mesh_test,
        intensity = y_test,
        colorscale = 'Reds',
        rot = rot,
        cmin = y_train.min(),
        cmax = y_train.max(),
    )
    fig.update_layout(scene_camera = dict(eye = dict(x = 1.1, y = 1.1, z = 0.2)))
    fig.write_html(f'{save_dir}bunny_signal_super-resolved_true.html')
    # fig.show()
    
    # print('Comparing predicted signal with true signal on the super-resolved mesh')
    # print(f'MSE Loss: {f.mse_loss(torch.Tensor(pred), torch.Tensor(y_test)).item():.6f}')
    # print(f'R2 Score: {r2_score(torch.Tensor(y_test).view(-1, model.inr_out_dim), torch.Tensor(pred).view(-1, model.inr_out_dim)):.6f}')
    # print(f'R2 Score: {model.r2_score(torch.Tensor(pred).view(-1, model.inr_out_dim), torch.Tensor(y_test).view(-1, model.inr_out_dim)):.6f}')

    
    print('Computing statistics for the super-resolved mesh')

    # Compute squared error per node
    mse = (y_test - pred) ** 2

    # Plot 8 - Plot error per node - super_resolved mesh only
    print('Plotting mse error for the super_resolved mesh')
    fig = draw_mesh(mesh_test, intensity = mse, colorscale = 'Reds', rot = rot)
    fig.update_layout(scene_camera = dict(eye = dict(x = 1.1, y = 1.1, z = 0.2)))
    fig.write_html(f'{save_dir}bunny_mse_loss_super-resolved.html')
    # fig.show()
    
    print(f'MSE Loss: {f.mse_loss(torch.Tensor(pred), torch.Tensor(y_test)).item()}')

    # Compute r2
    score = r2_score(y_test, pred[:])
    print(f'R2 score: {score}')

    # Compute r2 without 90th percentile outliers
    mask = mse < np.percentile(mse, 90)
    r2_score_adjusted = r2_score(y_test[mask], pred[:][mask])
    print(f'R2 score adjusted (90th percentile): {r2_score_adjusted}')

    # Compute r2 without 95th percentile outliers
    mask = mse < np.percentile(mse, 95)
    r2_score_adjusted = r2_score(y_test[mask], pred[:][mask])
    print(f'R2 score adjusted (95th percentile): {r2_score_adjusted}')

    # Plot distribution of squared error
    plt.figure(figsize = (2.2, 2.2))
    plt.hist(mse, bins = 10, density = True)
    plt.yscale('log')
    plt.xlabel('Squared error')
    plt.ylabel('Density')
    plt.tight_layout()
    plt.savefig(f'{save_dir}bunny_error_density_super-resolved.pdf', bbox_inches = 'tight')
    