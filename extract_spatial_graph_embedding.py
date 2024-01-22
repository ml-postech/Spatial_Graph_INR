from argparse import ArgumentParser

from src.spatial_embedding.utils.spatial_graph_embedding_extractor import Spatial_Graph_Embedding_Extractor
    

parser = ArgumentParser()
parser.add_argument('--model', default = '', type = str, help = 'Location of the model file to extract the embedding')
parser.add_argument('--dir', default = '', type = str, help = 'Directory where the model embedding information files are to be saved')
parser.add_argument('--name', default = '', type = str, help = 'Communal name of the files to be saved')

args = parser.parse_args()

Spatial_Graph_Embedding_Extractor(args.model, args.dir, args.name)
