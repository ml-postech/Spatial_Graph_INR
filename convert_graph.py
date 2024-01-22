from argparse import ArgumentParser

from src.utils.graph_converter import Graph_ZeroIndex_Start_Converter, Graph_OneIndex_Start_Converter



class InvalidConvertModeError(Exception):
    pass


# For files containing edge lists,
# PyMesh module, which is utilized for obtaining spatial graph embeddings, requires the starting index to be 0,
# while the codes utilized for Graph INR models require the starting index to be 1.
# Based on this, starting indexes should be adjusted properly.

parser = ArgumentParser()
parser.add_argument('--mode', default = '', type = str)
parser.add_argument('--graph_dir', default = 'data_generation/graph.txt', type = str)

args = parser.parse_args()

try:
    if args.mode == 'zero_start':
        Graph_ZeroIndex_Start_Converter(args.graph_dir)
        print('Graph successfully converted with index starting with 0')
    elif args.mode == 'one_start':
        Graph_OneIndex_Start_Converter(args.graph_dir)
        print('Graph successfully converted with index starting with 1')
    else:
        raise InvalidConvertModeError
except InvalidConvertModeError:
    print('Invalid graph convert mode. Enter one of the followings: zero_start or one_start.')
