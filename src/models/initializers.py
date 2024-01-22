import numpy as np
import torch

from torch import nn



# Initializer for MLP model when geometric_init == True
def geometric_initializer(layer, in_dim):
    nn.init.normal_(layer.weight, mean = np.sqrt(np.pi) / np.sqrt(in_dim), std = 0.00001)
    nn.init.constant_(layer.bias, -1)
    

# Initializer for MLP model with sine activation on only the first layer when geometric_init == False
def first_layer_sine_initializer(layer):
    with torch.no_grad():
        if hasattr(layer, 'weight'):
            num_input = layer.weight.size(-1)
            # See GINR paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            layer.weight.uniform_(-1 / num_input, 1 / num_input)


# Initializer for MLP model with sine activation on all layers when geometric_init == False
def sine_initializer(layer):
    with torch.no_grad():
        if hasattr(layer, 'weight'):
            num_input = layer.weight.size(-1)
            # See GINR supplement Sec. 1.5 for discussion of factor 30
            layer.weight.uniform_(-np.sqrt(6 / num_input) / 30, np.sqrt(6 / num_input) / 30)
