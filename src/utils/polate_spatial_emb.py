import torch

import numpy as np

from math import pi



# Obtain the dot product
def dot(x, y):
    return torch.sum(x * y, -1)


# Regularize the nonlast angle in spherical space to the range of [0, pi]
def sph_nonlast_angle_converter(theta):
    if theta < 0 or theta > (pi * 2):
        theta %= (pi * 2)
    if theta > pi:
        theta = (pi * 2) - theta
    return theta


# Obtain the weighted midpoint for the list of given points
# The dimension of point tensor should be 3
# Weighted midpoint is obtained via the summation of vectors with weights
def weighted_midpoint(points, weights):
    assert points.dim() == 3, 'Cannot get weighted midpoint - points tensor does not have dimension of 3'
    if weights.dim() == 1:
        # Shared weights
        assert points.size()[-2] == weights.size(0), 'Cannot get weighted midpoint - points tensor and weights tensor have different size'
        
        # Normalizing weight vector first
        weights = weights / torch.sum(weights, -1)
        weighted_sum = torch.zeros(points.size()[:-2] + points.size()[-1:])
        for i in range(weights.size(0)):
            weighted_sum += points[:, i, :] * weights[i]
        return weighted_sum
    elif weights.dim() == 2:
        # Different weights by groups of points
        assert points.size()[:-1] == weights.size(), 'Cannot get weighted midpoint - points tensor and weights tensor have different size'
        
        # Normalizing weight vector first
        weights = weights / torch.unsqueeze(torch.sum(weights, -1), -1)
        weighted_sum = torch.zeros(points.size()[:-2] + points.size()[-1:])
        for i in range(weights.size(0)):
            weighted_sum += points[:, i, :] * torch.unsqueeze(weights[:, i], -1)
        return weighted_sum
    else:
        raise ValueError('The dimension of weight tensor should be either 1 or 2')
    

# Obtain the weighted midpoint for given two points
# The dimension of point tensor should be 2
# The r values of both points are assumed to be equal (or slightly different due to float precision)
# For midpoints, certain angle can have two possible values,
# these angles are chosen as (smaller angle + larger angle) / 2
def sph_weighted_midpoint(points, t, norm = 0):
    assert points.dim() == 2, 'Cannot get weighted midpoint - points tensor does not have dimension of 2'
    assert points.size(0) == 2, 'Cannot get weighted midpoint - Two points should be given'
    sph_weighted_sum = torch.zeros(points.size(1))
    
    # Computing r value - if norm is given the r value is determined as the given norm value
    sph_weighted_sum[0] = points[0, 0] * t + points[1, 0] * (1 - t) if norm == 0 else norm
    
    # Computing angles except the last one - need to be reqularized to the range of [0, pi]
    for i in range(1, points.size(1) - 1):
        sph_weighted_sum[i] = sph_nonlast_angle_converter(points[0, i] * t + points[1, i] * (1 - t))
    
    # Computing the last angle
    # Ordering two angle values - 0 <= a <= b < pi * 2
    if points[0, -1] <= points[1, -1]:
        a = points[0, -1]
        b = points[1, -1]
        inverted = False
    else:
        a = points[1, -1]
        b = points[0, -1]
        inverted = True
    
    # Determining the direction of the polated point from a
    if b - a <= pi:
        direction = 'Positive'
    else:
        direction = 'Negative'
    
    # Computing the correct last angle value based on the determined direction
    # Applicable both for interpolating and extrapolating
    # Note that for the Negative case, the smaller arc always crosses the discontinuous point, so b - 2 * pi should be used instead of b
    if direction == 'Positive':
        if not inverted:
            sph_weighted_sum[-1] = (a * t + b * (1 - t)) % (pi * 2)
        else:
            sph_weighted_sum[-1] = (a * (1 - t) + b * t) % (pi * 2)
    elif direction == 'Negative':
        if not inverted:
            sph_weighted_sum[-1] = (a * t + (b - (pi * 2)) * (1 - t)) % (pi * 2)
        else:
            sph_weighted_sum[-1] = (a * (1 - t) + (b - (pi * 2)) * t) % (pi * 2)

    return sph_weighted_sum


# Projection from Minkowski hyperboloid (Lorentz) space to Klein space
def hyperboloid_to_klein(x):
    assert x.size(-1) > 1, 'Cannot be projected to Klein space'
    return x[..., 1:] / torch.unsqueeze(x[..., 0], -1)


# Projection from Klein space to Minkowski hyperboloid space
def klein_to_hyperboloid(x):
    one_size = x.size()[:-1] + (1, )
    one = torch.ones(one_size)
    t = torch.cat((one, x), dim = -1)
    return t / torch.unsqueeze(torch.sqrt(1 - dot(x, x)), -1)


# Convert Euclidean coordinates to spherical coordinates
# For the non-unique case - x_k, ... x_n are all zero when k is in the range of [2, n],
# the angles starting from (k - 1)-th will all be chosen to have zero values
def euclidean_to_spherical(x):
    x_sph = torch.zeros(x.size())
    dim = x.size(-1)
    
    # Compute r = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
    x_sph[..., 0] = torch.sqrt(dot(x, x))
    
    # Compute angles using atan2 functions - these angles range over [0, pi]
    for i in range(dim - 2):
        x_sph[..., i + 1] = torch.atan2(torch.sqrt(dot(x[..., i + 1:], x[..., i + 1:])), x[..., i])
        
    # Compute the last angle using atan2 function - this angle ranges over [0, pi * 2)
    x_sph[..., dim - 1] = (torch.atan2(x[..., dim - 1], x[..., dim - 2]) + (pi * 2)) % (pi * 2)
    return x_sph


# Convert spherical coordinates to Euclidean coordinates
def spherical_to_euclidean(x):
    x_euc = torch.ones(x.size())
    dim = x.size(-1)
    
    # Compute Euclidean coordinates using sine and cosine functions
    for i in range(dim):
        x_euc[..., i] *= x[..., 0]
        for j in range(1, i + 1):
            x_euc[..., i] *= torch.sin(x[..., j])
        if i < (dim - 1):
            x_euc[..., i] *= torch.cos(x[..., i + 1])
            
    # Threshold of absolute value for treating as zero-values, no threshold when the value is 0
    threshold = 1e-07
    x_euc[torch.abs(x_euc) < threshold] = 0
    return x_euc


# Interpolate or extrapolate in hyperbolic space with the formula
# Generalization of Einstein method is used - the formula u = tx + (1 - t)y is used in Klein space
# Inputs are assumed to be two 1D tensors for points and coefficient t
# Weighted Einstein midpoint
def polate_hyp(x, y, t: float):
    assert x.dim() == 1 and y.dim() == 1, 'Cannot polate in hyperbolic space - the dimension of both x and y should be 1'
    x_klein = hyperboloid_to_klein(torch.unsqueeze(x, 0))
    y_klein = hyperboloid_to_klein(torch.unsqueeze(y, 0))
    klein_points = torch.unsqueeze(torch.cat((x_klein, y_klein), dim = 0), 0)
    
    # Einstein midpoint
    klein_midpoint = weighted_midpoint(klein_points, torch.Tensor([t, 1 - t]))
    return torch.squeeze(klein_to_hyperboloid(klein_midpoint))


# Interpolate or extrapolate in spherical space with the formula
# Two points should not be on the opposite of the spherical space
# Inputs are assumed to be two 1D tensors for points and coefficient t
def polate_sph(x, y, t: float, norm = 0):
    assert x.dim() == 1 and y.dim() == 1, 'Cannot polate in spherical space - the dimension of both x and y should be 1'
    # General method for polating in spherical space
    # Coordinate conversions are required here
    # Weighted midpoint is obtained on spherical space, which is then converted back to Euclidean coordinate
    x_sph = torch.unsqueeze(euclidean_to_spherical(x), 0)
    y_sph = torch.unsqueeze(euclidean_to_spherical(y), 0)
    sph_points = torch.cat((x_sph, y_sph), dim = 0)
    sph_midpoint = sph_weighted_midpoint(sph_points, t, norm)
    return torch.squeeze(spherical_to_euclidean(sph_midpoint))

    # Midpoint-only case, obsolated here
    '''
    if t == 0.5 and not (x + y == 0).all():
        # Here, midpoint is much easier to compute - coordinate conversions are not required at all
        # First obtain the Euclidean midpoint u = (x + y) / 2, then project u on the spherical space
        # This only works when x and y are not on the exact opposite side of the sphere
        avg_norm = (torch.sqrt(dot(x, x)) + torch.sqrt(dot(y, y))) / 2 if norm == 0 else norm
        u = (x + y) / 2
        return u / torch.unsqueeze(torch.sqrt(dot(u, u)), -1) * avg_norm
    '''
            

# Interpolate or extrapolate in Euclidean space, with the formula tx + (1 - t)y
# Inputs are assumed to be two 1D tensors for points and coefficient t
def polate_euc(x, y, t: float):
    assert x.dim() == 1 and y.dim() == 1, 'Cannot polate in Euclidean space - the dimension of both x and y should be 1'
    return torch.squeeze(x * t + y * (1 - t))


# Obtain the midpoint of the geodesic segment in hyperbolc space
# Inputs are assumed to be two 1D tensors for points
# Einstein midpoint
def get_hyp_midpoint(x, y):
    assert x.dim() == 1 and y.dim() == 1, 'Cannot compute the midpoint in hyperbolic space - the dimension of both x and y should be 1'
    return polate_hyp(x, y, 0.5)


# Obtain the midpoint of the geodesic segment in spherical space
# Inputs are assumed to be two 1D tensors for points
def get_sph_midpoint(x, y, norm = 0):
    assert x.dim() == 1 and y.dim() == 1, 'Cannot compute the midpoint in spherical space - the dimension of both x and y should be 1'
    return polate_sph(x, y, 0.5, norm)


# Obtain the midpoint of the line segment in Euclidean space
# Inputs are assumed to be two 1D tensors for points
def get_euc_midpoint(x, y):
    assert x.dim() == 1 and y.dim() == 1, 'Cannot compute the midpoint in Euclidean space - the dimension of both x and y should be 1'
    return polate_euc(x, y, 0.5)


# Obtain the midpoint of the given two points with spatial embeddings
# Inputs are two 1D tensors for points and list with length 6 that contains the information of embedding
# Output is an 1D tensor for midpoint
def get_spatial_emb_midpoint(x, y, emb_info, norm = 0):
    assert len(emb_info) == 6, 'The list for embedding information should have 6 items with the order: hyp_dim, hyp_copy, sph_dim, sph_copy, euc_dim, euc_copy'
    spatial_midpoint = torch.Tensor()
    cur_idx = 0
    
    # Obtaining hyperbolic embeddings for the midpoint
    for _ in range(emb_info[1]):        
        spatial_midpoint = torch.cat((spatial_midpoint, get_hyp_midpoint(x[cur_idx:(cur_idx + emb_info[0] + 1)], y[cur_idx:(cur_idx + emb_info[0] + 1)])), dim = -1)
        cur_idx += (emb_info[0] + 1)
    
    # Obtaining spherical embeddings for the midpoint
    for _ in range(emb_info[3]):
        spatial_midpoint = torch.cat((spatial_midpoint, get_sph_midpoint(x[cur_idx:(cur_idx + emb_info[2] + 1)], y[cur_idx:(cur_idx + emb_info[2] + 1)], norm)), dim = -1)
        cur_idx += (emb_info[2] + 1)
    
    # Obtaining Euclidean embeddings for the midpoint
    for _ in range(emb_info[5]):
        spatial_midpoint = torch.cat((spatial_midpoint, get_euc_midpoint(x[cur_idx:(cur_idx + emb_info[4])], y[cur_idx:(cur_idx + emb_info[4])])), dim = -1)
        cur_idx += emb_info[4]

    return spatial_midpoint
