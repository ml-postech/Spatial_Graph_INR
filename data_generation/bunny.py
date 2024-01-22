import os, sys

import numpy as np
import pymesh

from argparse import ArgumentParser
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from src.models.core import parse_t_f
from src.plotting import figures
from src.utils.data_generation import (get_fourier, get_output_dir, load_mesh, mesh_to_graph, normalized_laplacian)



# Not a required function
'''
def graph_to_edge_list(adj, option):

    if option == 1:
        save_name = './data_generation/bunny/bunny_mesh.txt'
    else:
        save_name = f'./data_generation/bunny/bunny_mesh_res{option}.txt'
        
    file = open(save_name, 'w')
    
    adj_array = adj.toarray()
    for i in tqdm(range(len(adj_array)), total = len(adj_array), desc = 'Bunny mesh graph to edge list'):
        for j in range(i, len(adj_array)):
            if(adj_array[i, j] != 0):
                newline = str(i + 1) + ' ' + str(j + 1) + '\n'
                file.write(newline)
    
    print('Bunny mesh graph successfully converted to edge list format')
    print(f'Successfully saved as {save_name}')
    
    file.close()
'''


parser = ArgumentParser()
parser.add_argument('--res_option', default = 1, type = int)
parser.add_argument('--super_resolve', type = parse_t_f, default = False)

args = parser.parse_args()

if not args.res_option in [1, 2, 3, 4]:
    raise ValueError('Resolution option should be 1, 2, 3, or 4 (default = 1)')

# Load data
if args.res_option == 1:
    mesh = load_mesh('./data_generation/bunny/reconstruction/bun_zipper.ply')
else:
    mesh = load_mesh(f'./data_generation/bunny/reconstruction/bun_zipper_res{args.res_option}.ply')

points, edges, adj = mesh_to_graph(mesh)
n = points.shape[0]

if args.res_option <= 2 and args.super_resolve:
    save_name = f'./data_generation/bunny/bunny_res{args.res_option}_super_resolution/original_mesh.txt'
elif args.res_option == 1:
    save_name = './data_generation/bunny/bunny_mesh.txt'
else:
    save_name = f'./data_generation/bunny/bunny_mesh_res{args.res_option}.txt'
        
original_mesh_file = open(save_name, 'w')

print('Saving edge list for the original bunny mesh')

for i in tqdm(range(len(edges)), total = len(edges), desc = 'Exporting edge list'):
    newline = str(edges[i][0] + 1) + ' ' + str(edges[i][1] + 1) + '\n'
    original_mesh_file.write(newline)

print('Original bunny mesh graph successfully converted to edge list format')
print(f'Edge list successfully saved as {save_name}')

original_mesh_file.close()

if args.res_option <= 2 and args.super_resolve:
    original_mesh_points = points
    original_mesh_n_nodes = n
    original_mesh_faces = mesh.faces
    original_mesh_u = get_fourier(adj)
    
    mesh = pymesh.subdivide(mesh, order = 1)
    print('Obtaining target signals for the subdivided bunny mesh')
    
    points, edges, adj = mesh_to_graph(mesh)
    n = points.shape[0]
else:
    print('Obtaining target signals for the original bunny mesh')

# Target signal
# Evolve Gray-Scott reaction-diffusion model
Du, Dv, F, k = 0.16 * 4, 0.08 * 4, 0.060, 0.062
np.random.seed(1234)
lap = -normalized_laplacian(adj)  # Just for the diffusion
u = 0.2 * np.random.random(n) + 1
v = 0.2 * np.random.random(n)

n_iter = 30000  # Number of iterations for diffusion process
for i in tqdm(range(n_iter), desc = 'Obtaining target signals'):
    uvv = u * v * v
    u += Du * lap.dot(u) - uvv + F * (1 - u)
    v += Dv * lap.dot(v) + uvv - (F + k) * v

if args.res_option <= 2 and args.super_resolve:
    print('Target signals for the subdivided bunny mesh were successfully obtained')
else:
    print('Target signals for the original bunny mesh were successfully obtained')

# Plots
if args.res_option <= 2 and args.super_resolve:
    print('Plotting subdivided mesh with signals')
else:
    print('Plotting original mesh with signals')
    
rot = R.from_euler('xyz', [90, 00, 145], degrees = True).as_matrix()
fig = figures.draw_mesh(mesh, v, rot = rot, colorscale = 'Reds')
fig.show()

if args.res_option <= 2 and args.super_resolve:
    print('Plotting original mesh with signals')
    rot = R.from_euler('xyz', [90, 00, 145], degrees = True).as_matrix()
    fig = figures.draw_mesh(mesh, np.concatenate((v[:original_mesh_n_nodes], np.zeros((n - original_mesh_n_nodes,)))), rot = rot, colorscale = 'Reds')
    fig.show()

# Get Fourier features
u = get_fourier(adj)

if args.res_option <= 2 and args.super_resolve:
    original_output_dir = get_output_dir(f'bunny_res{args.res_option}/super_resolution/original/npz_files')
    original_npz_name = f'data_res{args.res_option}_original.npz'
    super_resolved_output_dir = get_output_dir(f'bunny_res{args.res_option}/super_resolution/super_resolved/npz_files')
    super_resolved_npz_name = f'data_res{args.res_option}_super_resolved.npz'
elif args.res_option == 1:
    output_dir = get_output_dir('bunny_v1/npz_files')
    npz_name = 'data.npz'
else:
    output_dir = get_output_dir(f'bunny_res{args.res_option}/npz_files')
    npz_name = f'data_res{args.res_option}.npz'

if args.res_option <= 2 and args.super_resolve:
    # npz file for the original bunny mesh
    np.savez(
        os.path.join(original_output_dir, original_npz_name),
        points = original_mesh_points,
        fourier = original_mesh_u,
        target = v[:original_mesh_n_nodes, None],
        faces = original_mesh_faces
    )
    # npz file for the super-resolved bunny mesh
    np.savez(
        os.path.join(super_resolved_output_dir, super_resolved_npz_name),
        points = points,
        fourier = u,
        target = v[:, None],
        faces = mesh.faces
    )
else:
    # npz file for the original bunny mesh
    np.savez(
        os.path.join(output_dir, npz_name),
        points = points,
        fourier = u,
        target = v[:, None],
        faces = mesh.faces
    )

if args.res_option <= 2 and args.super_resolve:
    print('Target signals for the subdivided bunny mesh were successfully obtained')
else:
    print('Target signals for the original bunny mesh were successfully obtained')
    