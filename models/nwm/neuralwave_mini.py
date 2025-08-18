"""
Author: Yang

Neural Wave (mini): A lightweight version of the Neural Wave model for proof of concept study.

This mini version implements:
- 1D wave propagation with refraction effects
- Simplified physics for directional wave refraction in varying depth
- Basic neural network integration for source term learning

Key features:
- One-dimensional spatial domain (along-shore or cross-shore)
- Wave refraction due to depth variations
- QUICKEST numerical scheme for wave advection
- UNet-based neural network for source term prediction
- Physics-AI hybrid approach for wave modeling

Note: This is a simplified version intended for proof of concept and research purposes.
"""

from typing import Optional, Tuple, Union, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import numpy as np

from .utils import wavenum_solver, get_Cg
import sys
sys.path.append('../')
from models.unet import UNet

class NeuralWave(nn.Module):
    """
    Neural Wave - Physics-AI Hybrid Model for wave prediction.
    """
    
    def __init__(
        self,
        lon: torch.Tensor,
        lat: torch.Tensor, 
        theta: torch.Tensor,
        freq: torch.Tensor,
        hypara_file: Optional[str] = None,
        device: str = 'cpu'
    ):
        super(NeuralWave, self).__init__()

        # Validate inputs
        assert theta.dim() == 1, f"theta should be 1D tensor, got shape {theta.shape}"
        assert freq.dim() == 1, f"freq should be 1D tensor, got shape {freq.shape}"

        self.lon = lon
        self.lat = lat
        self.theta = theta
        self.freq = freq
        
        self.device = device
        self.move_to_device()
        
        # Initialize spatial grid
        self._setup_spatial_grid()
        
        # Initialize spectral coordinates
        self._setup_spectral_grid()
        
        # Load hyperparameters
        if hypara_file:
            self.config = OmegaConf.load(hypara_file)
        else:
            self.config = self._get_default_config()

        # Initialize physics components
        self.physics_config = self.config.physics
            
        # Initialize neural network components
        self._setup_neural_networks()

        # Initialize model parameters
        self._setup_model_base()

        self.acti = lambda x: x
        self.mask = None
    
    def set_mask(
        self, 
        land_mask: torch.Tensor,
        bd_mask: torch.Tensor
    ) -> None:
        self.mask = ~torch.logical_or(bd_mask, land_mask)

    def _setup_model_base(self) -> None:
        """Initialize base model parameters."""
        self.dt_advec = self.physics_config.dt_advec
        self.radius_earth = self.physics_config.radius_earth
        self.FACTH = self.physics_config.FACTH
        self.FACVX = self.physics_config.FACVX
        self.FACHFA = self.physics_config.FACHFA

    def _setup_spatial_grid(self) -> None:
        """Setup spatial coordinate grids."""
        self.lon = self.lon.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        self.lat = self.lat.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        
        # Create extended grids for boundary calculations
        self.lonc = torch.cat([
            self.lon[:, -2:-1, :, :, :], 
            self.lon, 
            self.lon[:, 0:1, :, :, :]
        ], dim=1)
        
        self.latc = torch.cat([
            self.lat[:, :, -2:-1, :, :],
            self.lat,
            self.lat[:, :, 0:1, :, :]
        ], dim=2)
        
        # Calculate grid spacing
        self.dlon = torch.abs(0.5 * (
            torch.diff(self.lonc[:, 1:, :, :, :], dim=1) + 
            torch.diff(self.lonc[:, :-1, :, :, :], dim=1)
        ))
        
        self.dlat = torch.abs(0.5 * (
            torch.diff(self.latc[:, :, 1:, :, :], dim=2) + 
            torch.diff(self.latc[:, :, :-1, :, :], dim=2)
        ))
        
    def _setup_spectral_grid(self) -> None:
        """Setup spectral coordinate grids."""
        self.n_directions = len(self.theta)
        
        self.theta = self.theta.unsqueeze(-1).unsqueeze(0).unsqueeze(0).unsqueeze(0) 
        self.freq = self.freq.unsqueeze(0).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # Calculate angular spacing
        self.dtheta = torch.abs(self.theta[...,1:2,:] - self.theta[...,0:1,:]) if self.theta.shape[-2] > 1 else torch.tensor(2*np.pi)
        self.sinTheta = torch.round(torch.sin(self.theta) * 1e+5) * 1e-5
        self.cosTheta = torch.round(torch.cos(self.theta) * 1e+5) * 1e-5

        
    def _get_default_config(self) -> dict:
        """Get default model configuration."""
        config = {
            'neural_network': {
                'type': 'UNet',
                'params': {
                    'in_channels': 5,
                    'out_channels': 1,
                    'n_filters': 64,
                    'depth': 4
                }
            },
            'physics': {
                'enable_advection': True,
                'enable_refraction': True,
                'dt_advec': 0.01,  # Time step for advection
                'radius_earth': 6371000.0,  # Earth radius in meters
                'FACTH': 1.0,  # Factor for theta refraction
                'FACVX': 1.0,  # Factor for x advection
            }
        }
        return config
        
    def _setup_neural_networks(self) -> None:
        """Initialize neural network components."""
        nn_config = self.config.network
        
        if nn_config.type == 'UNet':
            self.network = UNet(
                n_channels=5,
                n_classes=1
            ).to(self.device)
        else:
            raise ValueError(f"Unknown neural network type: {nn_config.type}")
        
    def move_to_device(self) -> None:
        """Move all tensors to the specified device."""
        self.to(self.device)
        
        # Move coordinate tensors
        self.lon = self.lon.to(self.device)
        self.lat = self.lat.to(self.device) 
        self.theta = self.theta.to(self.device)
        self.freq = self.freq.to(self.device)
        
    def _process_inputs(
        self, 
        spectrum: torch.Tensor, 
        depth: torch.Tensor, 
        forcing: torch.Tensor
    ) -> None:
        """Validate input tensor shapes and types."""
        assert spectrum.dim() == 5, f"spectrum must be 5D, got {spectrum.dim()}D"
        assert depth.dim() == 3, f"depth must be 3D, got {depth.dim()}D"
        assert forcing.dim() == 5, f"forcing must be 5D, got {forcing.dim()}D"
        
        depth = depth.unsqueeze(-1).unsqueeze(-1)  # Ensure depth has shape [B, lon, lat, 1, 1]
        land_mask = depth > 0  # 1==sea, 0==land

        self.DdDx = torch.zeros_like(depth); self.DdDy = torch.zeros_like(depth)
        depthx = torch.cat((depth[:,-2:-1,:,:,:], depth, depth[:,0:1,:,:,:]), dim=1)
        depthy = torch.cat((depth[:,:,-2:-1,:,:], depth, depth[:,:,0:1,:,:]), dim=2)
        DdDx = 0.5 * (torch.diff(depthx[:,:-1,:,:,:], dim=1) + torch.diff(depthx[:,1:,:,:,:], dim=1)) / self.dlon
        DdDy = 0.5 * (torch.diff(depthy[:,:,:-1,:,:], dim=2) + torch.diff(depthy[:,:,1:,:,:], dim=2)) / self.dlat
        DdDx[~land_mask] = 0; DdDy[~land_mask] = 0

        self.DdDx = DdDx
        self.DdDy = DdDy

        return depth, land_mask
    
    def _prepare_physics(
        self,
        depth: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prepare physics parameters like wavenumber and group velocity."""
        # Calculate wavenumber using dispersion relation solver
        wavenum = torch.round(wavenum_solver(self.freq, depth), decimals=3).squeeze(-2)
        Cg = get_Cg(self.freq, wavenum, depth)
        # Handle numerical issues in wave physics calculations
        wavenum = torch.where(torch.isnan(wavenum), torch.zeros_like(wavenum), wavenum)
        Cg = torch.where(torch.isnan(Cg), torch.zeros_like(Cg), Cg)
        Cg = torch.where(torch.isinf(Cg), torch.full_like(Cg, 100.0), Cg)

        return wavenum, Cg
    
    def _getFlux_1order(self, N, Cg, land_mask, target_dim):
        """
        First order scheme.
        land_mask: 1==sea, 0==land
        """
        land2sea_mask = ~land_mask & torch.roll(land_mask,-1,dims=target_dim)
        Cg = torch.where(land2sea_mask, torch.roll(Cg,-1,dims=target_dim), Cg)
        Cg_b = 0.5 * (torch.roll(Cg, 1, dims=target_dim) + Cg)
        N_upwind = torch.where(Cg_b >= 0, torch.roll(N, 1, dims=target_dim), N) 
        FLN_minus = Cg_b * N_upwind
            
        return FLN_minus
    
    def _getFlux_3order(
        self,
        N: torch.Tensor,
        Cg: torch.Tensor,
        Dgrid: torch.Tensor,
        target_dim: int = 1,
        spatial_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        QUICKEST scheme with ULTIMATE TVD limiter for 4D wave spectrum, optimized for neural networks.
        See SUBROUTINE W3QCK3 in w3uqckmd.F90.
        """

        Dgrid = Dgrid.expand_as(N)
        # Calculate Cg_b (average of adjacent points along the specified target_dim)
        Cg_b = 0.5 * (torch.roll(Cg, 1, dims=target_dim) + Cg)

        # Calculate CFL number CFL
        CFL = (Cg_b * self.dt_advec) / Dgrid

        # Calculate upstream curvature CU
        CU = torch.zeros_like(N)
        mask_Cg_b_positive = Cg_b >= 0

        CU[mask_Cg_b_positive] = (torch.roll(N, 2, dims=target_dim) - 2 * torch.roll(N, 1, dims=target_dim) + N)[mask_Cg_b_positive] / (Dgrid[mask_Cg_b_positive]**2)
        CU[~mask_Cg_b_positive] = (torch.roll(N, 1, dims=target_dim) - 2 * N + torch.roll(N, -1, dims=target_dim))[~mask_Cg_b_positive] / (Dgrid[~mask_Cg_b_positive]**2)
        
        # Calculate Nb (flux at the left boundary)
        Nb = 0.5 * ((1 + CFL) * torch.roll(N.clone(), 1, dims=target_dim) + (1 - CFL) * N.clone()) - (1 - CFL**2) / 6 * CU * Dgrid**2

        N_c = torch.where(Cg_b >= 0, torch.roll(N, 1, dims=target_dim), N)
        N_u = torch.where(Cg_b >= 0, torch.roll(N, 2, dims=target_dim), torch.roll(N, -1, dims=target_dim))
        N_d = torch.where(Cg_b >= 0, N, torch.roll(N, 1, dims=target_dim))

        DQ = N_d - N_u  # downstream - upstream
        DQNZ = DQ.clone()
        DQNZ[(DQ <= 1e-15) & (DQ >= 0)] = 1e-15
        DQNZ[(DQ >= -1e-15) & (DQ <= 0)] = -1e-15
        
        Nc_tilde = (N_c - N_u) / DQNZ  # QCN
        Nc_tilde = torch.clamp(Nc_tilde, -0.1, 1.1)
        
        # Apply limiter conditions
        Nb_tilde = torch.maximum((Nb - N_u) / DQNZ, Nc_tilde)  # QBN
        
        Nclimit = Nc_tilde / torch.maximum(torch.tensor(1e-10), torch.abs(CFL))
        Nb_tilde = torch.minimum(Nb_tilde, torch.minimum(torch.tensor(1.0), Nclimit))
        
        Nb_rev = N_u + Nb_tilde * DQ
        CFAC = (2 * torch.abs(Nc_tilde - 0.5)).to(torch.int32)
        Nb = (1-CFAC) * Nb_rev + CFAC * N_c
        
        # if land_mask, use first order scheme here.
        if spatial_mask != None:
            neighbor_spatial_mask = (
                (spatial_mask == 0) | 
                (torch.roll(spatial_mask, 1, target_dim) == 0) | 
                (torch.roll(spatial_mask, -1, target_dim) == 0)
            )
            FLN_minus_1order = self._getFlux_1order(N, Cg, spatial_mask, target_dim)
            FLN_minus = torch.where(neighbor_spatial_mask, FLN_minus_1order, Cg_b * Nb)
            FLN_plus = torch.roll(FLN_minus, -1, dims=target_dim)
            sea2spatial_mask = spatial_mask & ~torch.roll(spatial_mask,-1,dims=target_dim) & (Cg>=0)
            FLN_plus = torch.where(sea2spatial_mask, Cg * N, FLN_plus)
        else:
            # flux at the left boundary (Fi,-)
            FLN_minus = Cg_b * Nb
            FLN_plus = torch.roll(FLN_minus, -1, dims=target_dim)

        return FLN_minus, FLN_plus
    
    def _apply_advection(
        self, 
        spectrum: torch.Tensor,
        Cg: torch.tensor,
        land_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply wave advection physics."""
        spec = self.acti(self.FACHFA) * spectrum.clone()
        spec = torch.where(torch.isnan(spec), torch.zeros_like(spec), spec)
        spec = torch.where(torch.isinf(spec), torch.zeros_like(spec), spec)
        lamda = self.FACVX * Cg * self.sinTheta
        FLN_negx, FLN_posx = self._getFlux_3order(spec, lamda, self.dlon, target_dim=1, spatial_mask=land_mask)
        xPro = (FLN_negx-FLN_posx) / self.dlon
        return xPro

    def _apply_refraction(
        self,
        spectrum: torch.Tensor,
        depth: torch.Tensor,
        Cg: torch.Tensor,
        wavenum: torch.Tensor
    ) -> torch.Tensor:
        """Apply wave refraction physics.""" 
        mk_dsdd = depth * wavenum >= 5
        # Partial derivative of sigma for depth.
        DsDd = (Cg * wavenum - 0.5 * self.freq * 2 * torch.pi) / depth
        DsDd[torch.isnan(DsDd)]=0
        DsDd[mk_dsdd] = 0; DsDd[DsDd < 0] = 0
        assert DsDd.min() >= 0, 'DsDd should be positive. See line 1047 in w3pro1md.F90'
        VCTH = self.FACTH * torch.pow(wavenum, -1) * (DsDd * (self.cosTheta * self.DdDx - self.sinTheta * self.DdDy))  
        VCTH[torch.isnan(VCTH)]=0; VCTH[torch.isinf(VCTH)]=0
        FLN_negth, FLN_posth = self._getFlux_3order(spectrum, VCTH, Dgrid=self.dtheta, target_dim=3)
        thetaPro = (FLN_negth - FLN_posth) / self.dtheta
        assert FLN_posth[:,:,:,-1].sum() == FLN_negth[:,:,:,0].sum(), f'FLN_posth[:,:,:,-1].sum() = {FLN_posth[:,:,:,-1].sum()}, FLN_negth[:,:,:,0].sum() = {FLN_negth[:,:,:,0].sum()}'
        return thetaPro[:,1:-1,1:-1,:,:]
        
    def _apply_boundary_conditions(
        self,
        spectrum: torch.Tensor,
        boundary_condition: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Apply boundary conditions to the spectrum."""
        # Simple implementation: use boundary values where specified
        if boundary_condition is None:
            bd_mask = torch.ones(spectrum.shape, dtype=torch.bool, device=spectrum.device)
        else:
            bd_mask = (boundary_condition.sum(dim=(-2, -1), keepdim=True) > 0).expand_as(spectrum)
        # Set boundary (outer ring) to zero while preserving inner values
        spectrum_inner = spectrum[:, 1:-1, 1:-1, :, :].clone()
        spectrum.zero_()
        spectrum[:, 1:-1, 1:-1, :, :] = spectrum_inner
        spectrum = torch.where(bd_mask, boundary_condition, spectrum)
        return spectrum, bd_mask
    
    def _apply_land_mask(
        self,
        spectrum: torch.Tensor,
        land_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply land mask to the spectrum."""
        land_mask = land_mask.expand_as(spectrum)
        spectrum = torch.where(land_mask, spectrum, torch.zeros_like(spectrum))
        return spectrum, land_mask

    def _learnFieldStot(
        self, 
        spectrum: torch.Tensor,
        forcings: torch.Tensor
    ) -> torch.Tensor:
        """
        Using the neural network to predict the Field-Stot.
        No training process in this function !!!

        ATTENTION:
        Func learnFieldStot only fit for 1-dimension Case.
        For example:
        spectrum -> [b, w, l, th, freq]
        can be simplified as -> [b, 1, w, th, 1], as l & freq are const.

        DataFlow: 
        spectrum [b, w, l, th, freq] -> spec_input [b, 1, w, th, 1]
                                                    |
                                                    | --> cnn_input [b, c+1, w, th]
                                                    |                 |
        forcings_input [b, 1, w, th, c] ------------|                 |
                                                                     UNet
                                                                      |
                                                                      |
        stot_set [b, w, l, th, freq] <----------------- stot_output [b, 1, w, th]
        +
        spectrum [b, w, l, th, freq] -> spectrum_{t+1}
        """

        stot = torch.zeros_like(spectrum, device=self.device)
        spec_input = spectrum[:,:,1:-1,:,1:-1].permute(0, 2, 1, 3, 4)
        cnn_input = torch.cat([spec_input, forcings], dim=-1).permute(0,4,2,3,1)

        stot_output = self.network(cnn_input.squeeze(-1)).unsqueeze(-1)
        stot_output = stot_output.permute(0, 2, 1, 3, 4)
        stot[:, :, 1:-1, :, 1:-1] = stot_output

        return stot

    @classmethod
    def load_pretrained(
        cls, 
        model_path: str,
        lon: torch.Tensor,
        lat: torch.Tensor,
        theta: torch.Tensor, 
        freq: torch.Tensor,
        hypara_file: Optional[str] = None,
        device: str = 'cpu'
    ) -> 'NeuralWave':
        """
        Load a pre-trained model from file.
        """
        model = cls(lon, lat, theta, freq, hypara_file, device=device)
        state_dict = torch.load(model_path, map_location=device)
        model.network.load_state_dict(state_dict)
        model.eval()
        return model

    def save_model(self, save_path: str) -> None:
        """Save model state dict to file."""
        torch.save(self.network.state_dict(), save_path)

    def get_model_info(self) -> dict:
        """Get model configuration and parameter information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'device': self.device,
            'grid_shape': {
                'longitude': self.lon.shape[1],
                'latitude': self.lat.shape[2], 
                'directions': self.n_directions,
                'frequencies': self.n_frequencies
            },
            'config': OmegaConf.to_yaml(self.config)
        }
    
    def forward(
        self,
        spectrum: torch.Tensor,
        depth: torch.Tensor,
        forcing: torch.Tensor,
        boundary_condition: Optional[torch.Tensor] = None,
        dynamicPro: bool = True,
        learnStot: bool = True
    ) -> torch.Tensor:
        """
        Forward pass of the neural wave model.
        """

        # Input validation
        depth, land_mask = self._process_inputs(spectrum, depth, forcing)

        # Prepare physics parameters
        wavenum, Cg = self._prepare_physics(depth)

        if dynamicPro:
            if self.physics_config.enable_refraction:
                thetaPro = self._apply_refraction(spectrum, depth, Cg, wavenum)
                spectrum[:, 1:-1, 1:-1, :, :] += 0.5 * self.dt_advec * thetaPro
                spectrum = torch.clamp(spectrum, min=0.0)
                spectrum, _ = self._apply_land_mask(spectrum, land_mask)
            
            spectrum, _ = self._apply_boundary_conditions(spectrum, boundary_condition)
            
            if self.physics_config.enable_advection:
                xPro = self._apply_advection(spectrum, Cg, land_mask)
                spectrum += self.dt_advec * xPro
                spectrum = torch.clamp(spectrum, min=0.0)
                spectrum, _ = self._apply_land_mask(spectrum, land_mask)
            
            spectrum, bd_mask = self._apply_boundary_conditions(spectrum, boundary_condition)

            if self.physics_config.enable_refraction:
                thetaPro = self._apply_refraction(spectrum, depth, Cg, wavenum)
                spectrum[:, 1:-1, 1:-1, :, :] += 0.5 * self.dt_advec * thetaPro
                spectrum = torch.clamp(spectrum, min=0.0)
                spectrum, land_mask = self._apply_land_mask(spectrum, land_mask)
        
        # Neural network prediction
        if learnStot:
            stot = self._learnFieldStot(spectrum, forcing)
            spectrum = spectrum + stot
            spectrum, bd_mask = self._apply_boundary_conditions(spectrum, boundary_condition)
            spectrum[spectrum < 0] = 0
        
        self.set_mask(land_mask, bd_mask)

        return spectrum