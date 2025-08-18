# -*- coding: utf-8 -*-
"""
@author: Yang
"""

import torch
import numpy as np
import random

def seed_everything(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed

def stack_tensors(*args):
    num_tensors = len(args)
    if num_tensors == 0:
        raise ValueError("The amount of input-vars at least should be 1.")
    tensor_shape = args[0].shape
    for tensor in args:
        if tensor.shape != tensor_shape:
            raise ValueError("The input-vars should have the same shape.")
    stack_vars = torch.stack(args, dim=-1)
    return stack_vars

def series_to_superised_out_4d(var2superised, Lag):
    B,X,Y,C=var2superised.shape
    if Lag>1:
        ncols = int(B - Lag + 1)  # lower bound are 0 .
        resAll_out = np.zeros((ncols, Lag, X, Y, C), dtype=float)
        for ii in range(ncols):
            resAll_out[ii] = var2superised[ii:ii+Lag]
    else:
        resAll_out=var2superised[:,np.newaxis,:,:,:]
    return resAll_out

def series_to_superised_out_5d(var2superised, Lag):
    B, X, Y, Ntheta, Nk = var2superised.shape
    if Lag > 1:
        ncols = int(B - Lag + 1)  # lower bound are 0.
        resAll_out = np.zeros((ncols, Lag, X, Y, Ntheta, Nk), dtype=float)
        for ii in range(ncols):
            resAll_out[ii] = var2superised[ii:ii+Lag]
    else:
        resAll_out = var2superised[:, np.newaxis, :, :, :, :]
    return resAll_out

def generate_sparse_mask(shape, sparsity):
    '''
    More sparse, more zero.
    '''
    num_elements = np.prod(shape)
    num_ones = int((1 - sparsity) * num_elements)  
    num_zeros = num_elements - num_ones

    mask = np.array([1] * num_ones + [0] * num_zeros)
    np.random.seed(42)
    np.random.shuffle(mask)
    
    return mask.reshape(shape)

def beji_wavenumber(freq, waterdepth):
    """
    Use Beji's iterative method to calculate the wavenumber of the wave field for a 3D tensor of frequencies.
    
    Parameters:
    freq (torch.Tensor): Wave frequency (Hz) tensor with shape (batch_size, channels, num_frequencies)
    waterdepth (float): Water depth (m)
    
    Returns:
    k (torch.Tensor): Wavenumber (1/m) tensor with the same shape as freq
    """
    # Constants
    g = 9.81  # gravity acceleration (m/s^2)
    omega = 2 * torch.pi * freq  # angular frequency (rad/s)

    # Initial guess for wavenumber using deep water approximation
    k = omega**2 / g

    # Iteratively refine wavenumber using Beji's method
    tol = 1e-6  # tolerance for convergence
    max_iter = 1000  # maximum iterations
    for i in range(max_iter):
        # Calculate the next wavenumber using the iterative formula
        k_next = omega**2 / (g * torch.tanh(k * waterdepth))

        # Check for convergence
        if torch.all(torch.abs(k_next - k) < tol):
            break

        # Update k for the next iteration
        k = k_next
        
    # close boundary for land-area
    k[torch.isnan(k)] = 0
    return k

def dispersion_residual(freq, depth, k):
    """
    Calculate the residual of the dispersion relation.
    
    Parameters:
    freq (torch.Tensor): Wave frequency (Hz) tensor, shape [B, lon, lat, fir, freq]
    depth (torch.Tensor): Water depth (m), shape [B, lon, lat, 1, 1]
    k (torch.Tensor): Wavenumber tensor (1/m), shape same as freq
    
    Returns:
    torch.Tensor: Residual of the dispersion relation, shape same as freq
    """
    # Constants
    g = 9.81  # gravity acceleration (m/s^2)
    omega = 2 * torch.pi * freq  # angular frequency (rad/s)
    depth = depth.unsqueeze(-1)  # Ensure correct shape for broadcasting
    
    # Calculate the residual of the dispersion relation
    residual = torch.abs(torch.tanh(k * depth) - (omega**2) / (g * k))
    return residual

def wavenum_solver(freq, depth, max_iter=50, tol=1e-6):
    """
    By YGY
    
    Optimized wave number solver using binary search.
    
    Parameters:
    freq (torch.Tensor): Wave frequency (Hz) tensor, shape [B, lon, lat, fir, freq]
    depth (torch.Tensor): Water depth (m), shape [B, lon, lat, 1, 1]
    max_iter (int): Maximum number of iterations for the solver
    tol (float): Convergence tolerance for the wave number solution
    
    Returns:
    torch.Tensor: Wave number tensor, shape same as freq
    """
    g = 9.806  # gravity acceleration (m/s^2)
    omega = 2 * torch.pi * freq  # angular frequency (rad/s)
    depth = depth.unsqueeze(-1)
    
    # Initialize bounds for k
    k_lower = torch.full_like(freq, 1e-4)
    k_upper = torch.full_like(freq, 10.0)
    
    for _ in range(max_iter):
        k_mid = (k_lower + k_upper) / 2
        kh_mid = k_mid * depth
        f_mid = torch.tanh(kh_mid) - (omega**2 * depth) / (g * kh_mid)
        
        k_upper = torch.where(f_mid > 0, k_mid, k_upper)
        k_lower = torch.where(f_mid <= 0, k_mid, k_lower)
        
        if torch.max(torch.abs(k_upper - k_lower)) < tol:
            break
    
    k = (k_lower + k_upper) / 2
    k[torch.isnan(k)] = 0
    return k


def get_Cg(
    freq: torch.Tensor,
    wavenum: torch.Tensor,
    depth: torch.Tensor
) -> torch.Tensor:
    """
    Calculate group velocity.
    
    Parameters
    ----------
    freq : torch.Tensor
        Wave frequency [Hz]
    depth : torch.Tensor
        Water depth [m] 
    g : float, default=9.81
        Gravitational acceleration [m/s²]
        
    Returns
    -------
    torch.Tensor
        Group velocity [m/s]
    """
    sigma = 2 * torch.pi * freq 
    tt = torch.sinh(2 * wavenum * depth)
    n = 1/2 + (wavenum * depth) / torch.sinh(2 * wavenum * depth) 
    Cg = n * (sigma / wavenum)
    # Cg = freq / wavenum
    return Cg

def get_integral_m(spectrum_distribution, df, dtheta, frequencies, order: int):
    return torch.sum(spectrum_distribution * df * dtheta * (frequencies ** order), dim=(3, 4))

def get_wave_intparams(spectra, theta, freq):
    """
    Get the wave integral parameters from the wave spectra.
    SWH, MWD
    spectra: [B, leadtime, lon+2, lat+2, Ntheta, Nk]
    """
    
    directions = theta[None, None, None, None, :, None].to(spectra.device)
    dtheta = torch.deg2rad(torch.abs(theta[0] - theta[1])).to(spectra.device)
    frequencies = freq[None, None, None, None, None, :].to(spectra.device)
    df = torch.diff(frequencies, dim=-1, prepend=frequencies[:, :, :, :, :, :1]).to(spectra.device)

    m_0 = get_integral_m(spectra, df, dtheta, frequencies, order=0)
    SWH = 4 * torch.sqrt(m_0 + 1e-16)

    directions_rad = torch.deg2rad(directions)
    SF = torch.nansum(torch.sin(directions_rad) * spectra * dtheta * df, dim=(4, 5))
    CF = torch.nansum(torch.cos(directions_rad) * spectra * dtheta * df, dim=(4, 5))
    MWD = torch.rad2deg(torch.atan2(SF + 1e-16, CF + 1e-16) + torch.pi)

    return SWH, MWD

if __name__ == '__main__':
    # [B, lon, lat, dir, freq]
    # Example usage
    freq = torch.rand(1, 1, 1, 1, 30)  # 3D tensor of frequencies
    waterdepth = 50.0  * torch.rand(1, 10, 10, 1, 1)
    # k = wavenum_solver(freq, waterdepth)
    k = beji_wavenumber(freq, waterdepth)

    residual = dispersion_residual(freq, waterdepth, k.unsqueeze(-2))
    # print("Residual:", residual)
    print("Max Residual:", torch.max(residual).item())
    print("Mean Residual:", torch.mean(residual).item())