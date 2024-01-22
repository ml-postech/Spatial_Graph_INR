import os

import torch
import numpy as np

from src.spatial_embedding.multicurvature_models import ProductEmbedding
from src.spatial_embedding.multicurvature_parameter import RParameter


class NoNameError(Exception):
    pass

class DirectoryNotFoundError(Exception):
    pass


# Receives a model file to be loaded, whose embedding is to be extracted
# If both the directory and the communal name are given,
# The extracted model embedding information is saved based on the given arguments
def Spatial_Graph_Embedding_Extractor(model, dir, name):
    try:
        if(os.path.isfile(model) == False):
            raise FileNotFoundError
        
        m = torch.load(model).to('cpu')
        
        hyp_embedding = None
        hyp_dim = 0
        hyp_copy = len(m.hyp_params)
        sph_embedding = None
        sph_dim = 0
        sph_copy = len(m.sph_params)
        euc_embedding = None
        euc_dim = 0
        euc_copy = len(m.euc_params)
        
        if hyp_copy > 0:
            (n_edges, hyp_dim) = m.hyp_params[0].detach().numpy().shape
            hyp_dim -= 1
            hyp_embedding = np.empty((n_edges, 0))
            
            for i in range(hyp_copy):
                # Need to determine the meaning of the first value of each row
                # hyp_embedding = np.append(hyp_embedding, m.hyp_params[i].detach().numpy()[:, 1:], axis = 1)
                hyp_embedding = np.append(hyp_embedding, m.hyp_params[i].detach().numpy(), axis = 1)
        
        if sph_copy > 0:
            (n_edges, sph_dim) = m.sph_params[0].detach().numpy().shape
            sph_dim -= 1
            sph_embedding = np.empty((n_edges, 0))
            
            for i in range(sph_copy):
                # Need to determine the meaning of the first value of each row
                # sph_embedding = np.append(sph_embedding, m.sph_params[i].detach().numpy()[:, 1:], axis = 1)
                sph_embedding = np.append(sph_embedding, m.sph_params[i].detach().numpy(), axis = 1)
        
        if euc_copy > 0:
            (n_edges, euc_dim) = m.euc_params[0].detach().numpy().shape
            euc_embedding = np.empty((n_edges, 0))
            
            for i in range(euc_copy):
                euc_embedding = np.append(euc_embedding, m.euc_params[i].detach().numpy(), axis = 1)
        
        print('Model embedding information was successfully extracted!')
        
        hyp_emb_str = 'Hyperbolic embedding - '
        if hyp_copy > 0:
            hyp_emb_str += f'Dimension: {hyp_dim}, Number of copies: {hyp_copy}'
        else:
            hyp_emb_str += 'None'
            
        sph_emb_str = 'Spherical embedding  - '
        if sph_copy > 0:
            sph_emb_str += f'Dimension: {sph_dim}, Number of copies: {sph_copy}'
        else:
            sph_emb_str += 'None'
            
        euc_emb_str = 'Euclidean embedding  - '
        if euc_copy > 0:
            euc_emb_str += f'Dimension: {euc_dim}, Number of copies: {euc_copy}'
        else:
            euc_emb_str += 'None'
        
        if dir is not None:
            if name is None:
                raise NoNameError
            if not os.path.isdir(dir):
                raise DirectoryNotFoundError
            
            emb_info_file = open(f'{dir}/{name}_emb_info.txt', 'w')
            
            emb_info_file.write('              Dim     Copy\n')
            emb_info_file.write(f'Hyperbolic:   {hyp_dim}       {hyp_copy}\n')
            emb_info_file.write(f'Spherical:    {sph_dim}       {sph_copy}\n')
            emb_info_file.write(f'Euclidean:    {euc_dim}       {euc_copy}\n')
            
            emb_info_file.close
            
            np.savez(f'{dir}/{name}_emb.npz', hyp = hyp_embedding, sph = sph_embedding, euc = euc_embedding)
            
            print('Extracted model embedding information was successfully saved.') 
        elif name is not None:
            print('Enter both the directory and the communal file name to save the extracted model embedding info.')
        else:
            print('Extracted model embedding information was not saved.')
            
        
    except FileNotFoundError:
        print('Model file was not found.')
    except NoNameError:
        print('Communal file name was not given.')
    except DirectoryNotFoundError:
        print('Directory was not found.')
        