# Perform mesh subdivision with simple method in PyMesh package, with node features of spatial graph embeddings
import numpy as np
import torch

from tqdm import tqdm

from src.utils.data_generation import load_mesh
from src.utils.load_emb_file import load_emb_info
from src.utils.polate_spatial_emb import get_spatial_emb_midpoint, get_euc_midpoint



# Mesh subdivision for Spatial Graph INR
def spatial_mesh_simple_subdivision(mesh, embs, emb_dir, emb_name):
    mesh_load = load_mesh(mesh)

    # Add new nodes and their corresponding embeddings in the subdivided mesh
    # The order of new nodes are identical to that of subdivided mesh created with PyMesh package
    # Node embeddings are concatenated with the order of hyp, sph, and euc
    hyp_dim, hyp_copy, sph_dim, sph_copy, euc_dim, euc_copy = load_emb_info(emb_dir, emb_name)
    assert ((hyp_dim + 1) * hyp_copy) + ((sph_dim + 1) * sph_copy) + (euc_dim * euc_copy) == np.shape(embs)[1], 'Embedding dimension does not match with given embedding information'
    emb_info = [hyp_dim, hyp_copy, sph_dim, sph_copy, euc_dim, euc_copy]

    new_nodes = np.empty((0, np.shape(embs)[1]))
    
    for i in tqdm(range(len(mesh_load.faces)), total = len(mesh_load.faces), desc = 'Subdividing mesh with spatial graph embeddings'):
        n1_idx = mesh_load.faces[i, 0]
        n2_idx = mesh_load.faces[i, 1]
        n3_idx = mesh_load.faces[i, 2]
        
        n1 = torch.Tensor(embs[n1_idx])
        n2 = torch.Tensor(embs[n2_idx])
        n3 = torch.Tensor(embs[n3_idx])
        
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_spatial_emb_midpoint(n1, n2, emb_info, norm = 1.0), dim = 0).numpy(), axis = 0)
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_spatial_emb_midpoint(n2, n3, emb_info, norm = 1.0), dim = 0).numpy(), axis = 0)
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_spatial_emb_midpoint(n3, n1, emb_info, norm = 1.0), dim = 0).numpy(), axis = 0)
        
    original_nodes = embs

    # Removing duplicate nodes - each new nodes appear twice
    new_nodes, idx = np.unique(np.array(new_nodes), axis = 0, return_index = True)
    new_nodes = new_nodes[np.argsort(idx)]
    subdivided_mesh_nodes = np.concatenate([original_nodes, new_nodes], axis = 0)

    print('Mesh subdivision with simple method and order = 1 with spatial graph embeddings has been completed')

    return subdivided_mesh_nodes


# Mesh subdivision for INR
def euc_mesh_simple_subdivision(mesh, points):
    mesh_load = load_mesh(mesh)
    
    new_nodes = np.empty((0, np.shape(points)[1]))
    
    for i in tqdm(range(len(mesh_load.faces)), total = len(mesh_load.faces), desc = 'Subdividing mesh with Euclidean coordinates'):
        n1_idx = mesh_load.faces[i, 0]
        n2_idx = mesh_load.faces[i, 1]
        n3_idx = mesh_load.faces[i, 2]
        
        n1 = torch.Tensor(points[n1_idx])
        n2 = torch.Tensor(points[n2_idx])
        n3 = torch.Tensor(points[n3_idx])
        
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_euc_midpoint(n1, n2), dim = 0).numpy(), axis = 0)
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_euc_midpoint(n2, n3), dim = 0).numpy(), axis = 0)
        new_nodes = np.append(new_nodes, torch.unsqueeze(get_euc_midpoint(n3, n1), dim = 0).numpy(), axis = 0)
        
    original_nodes = points
    
    # Removing duplicate nodes - each new nodes appear twice
    new_nodes, idx = np.unique(np.array(new_nodes), axis = 0, return_index = True)
    new_nodes = new_nodes[np.argsort(idx)]
    subdivided_mesh_nodes = np.concatenate([original_nodes, new_nodes], axis = 0)

    print('Mesh subdivision with simple method and order = 1 with Euclidean coordinates has been completed')

    return subdivided_mesh_nodes