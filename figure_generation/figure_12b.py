# -*- coding: utf-8 -*-
"""
@author: Yang
"""

import sys
sys.path.append('..')
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # Avoid KMP duplicate lib error
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
from scipy import ndimage
from matplotlib.font_manager import FontProperties
import netCDF4 as nc
import torch

# Set global font to Times New Roman
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']

# plot_scatter 函数保持不变，以保留原始绘图风格
def plot_scatter(data_type, pred, true, bins=120, sigma=10.0, cmap='magma_r', alpha=1, s=10,
                 ax=None, fig=None, subplot_title_prefix="", save_path_standalone=None):
    """
    Draws a scatter plot of predicted vs. true values, with Gaussian smoothed density,
    statistical information (including MRE), and a colorbar. Can draw on a given Axes or create a new figure.
    """
    is_subplot_mode = (ax is not None and fig is not None)

    # Flatten arrays
    pred_flat = pred.flatten()
    true_flat = true.flatten()

    # Calculate statistics
    n_points = pred_flat.size
    rmse = np.sqrt(np.mean((true_flat - pred_flat)**2))
    bias = np.mean(pred_flat - true_flat)
    
    # calculate CC
    mean_true = np.mean(true_flat)
    ss_tot = np.sum((true_flat - mean_true)**2)
    ss_res = np.sum((true_flat - pred_flat)**2)
    CC = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    # Calculate Mean Relative Error (MRE)
    # Avoid division by zero by only considering non-zero true values
    mask_nonzero_true = true_flat != 0
    relative_errors = np.abs(pred_flat[mask_nonzero_true] - true_flat[mask_nonzero_true]) / true_flat[mask_nonzero_true]
    mre = np.mean(relative_errors) * 100  # As percentage
    mre_text = f"{mre:.2f} %"

    fontsize_axes_ticks = 14
    figsize_standalone = (5.5, 5)

    if data_type == 'swh':
        xlabel = 'True SWH (m)'
        ylabel = 'Predicted SWH (m)'
        min_val = 0.0
        max_val = 1.0
        margin = 0
        axis_min = min_val - margin
        axis_max = max_val + margin
        unit_rmse = 'm'
        unit_bias = 'm'
    elif data_type == 'mwd':
        xlabel = 'True MWD (°)'
        ylabel = 'Predicted MWD (°)'
        axis_min = 220
        axis_max = 320
        unit_rmse = '°'
        unit_bias = '°'
    else:
        raise ValueError("data_type must be 'swh' or 'mwd'")

    current_ax = ax if is_subplot_mode else plt.subplots(figsize=figsize_standalone, dpi=100)[1]
    current_fig = fig if is_subplot_mode else current_ax.get_figure()

    H, xe, ye = np.histogram2d(true_flat, pred_flat,
                               bins=bins,
                               range=[[axis_min, axis_max], [axis_min, axis_max]])
    sH = ndimage.gaussian_filter(H, sigma=sigma, order=0, mode='constant', cval=0.0)
    H = H.T
    sH = sH.T
    Hind = np.ravel(H)

    xc = (xe[:-1] + xe[1:]) / 2.0
    yc = (ye[:-1] + ye[1:]) / 2.0
    xv, yv = np.meshgrid(xc, yc)

    x_scatter = np.ravel(xv)[Hind != 0]
    y_scatter = np.ravel(yv)[Hind != 0]
    z_density = np.ravel(sH)[Hind != 0]

    non_zero_mask = z_density != 0
    x_scatter_nz = x_scatter[non_zero_mask]
    y_scatter_nz = y_scatter[non_zero_mask]
    z_density_nz = z_density[non_zero_mask]
    
    if len(z_density_nz) > 1 and z_density_nz.max() != z_density_nz.min():
        z_norm = (z_density_nz - z_density_nz.min()) / (z_density_nz.max() - z_density_nz.min())
    elif len(z_density_nz) > 0:
        z_norm = np.ones_like(z_density_nz) * 0.5
    else:
        z_norm = np.array([])

    if len(x_scatter_nz) > 0:
        scatter_plot = current_ax.scatter(x_scatter_nz, y_scatter_nz,
                                          edgecolor='none', c=z_norm, cmap=cmap, alpha=alpha, s=s)
        cbar = current_fig.colorbar(scatter_plot, ax=current_ax)
        cbar.set_label('Density', fontproperties=FontProperties(family='Times New Roman', size=14))
        cbar.ax.tick_params(labelsize=14)
        for label in cbar.ax.get_yticklabels():
            label.set_fontproperties(FontProperties(family='Times New Roman', size=10))
    else:
        current_ax.text(0.5, 0.5, "No data to plot", horizontalalignment='center', verticalalignment='center', transform=current_ax.transAxes)

    current_ax.set_xlim(axis_min, axis_max)
    current_ax.set_ylim(axis_min, axis_max)
    current_ax.spines['top'].set_linewidth(2)
    current_ax.spines['bottom'].set_linewidth(2)
    current_ax.spines['left'].set_linewidth(2)
    current_ax.spines['right'].set_linewidth(2)
    current_ax.grid(True, linestyle='--', alpha=0.7)

    font_axes = FontProperties(family='Times New Roman', size=fontsize_axes_ticks)
    num_ticks = 11
    current_ax.set_xticks(np.linspace(axis_min, axis_max, num_ticks))
    current_ax.set_yticks(np.linspace(axis_min, axis_max, num_ticks))

    if data_type == 'swh':
        current_ax.set_xticklabels([f"{x:.1f}" for x in np.linspace(axis_min, axis_max, num_ticks)], fontproperties=font_axes)
        current_ax.set_yticklabels([f"{y:.1f}" for y in np.linspace(axis_min, axis_max, num_ticks)], fontproperties=font_axes)
    elif data_type == 'mwd':
        current_ax.set_xticklabels([f"{int(x)}" for x in np.linspace(axis_min, axis_max, num_ticks)], fontproperties=font_axes)
        current_ax.set_yticklabels([f"{int(y)}" for y in np.linspace(axis_min, axis_max, num_ticks)], fontproperties=font_axes)

    current_ax.set_xlabel(xlabel, fontproperties=font_axes)
    current_ax.set_ylabel(ylabel, fontproperties=font_axes)

    # current_ax.set_title(plot_title_text, fontproperties=FontProperties(family='Times New Roman', size=title_fontsize, weight=title_fontweight))

    current_ax.plot(np.linspace(axis_min, axis_max, 100), np.linspace(axis_min, axis_max, 100),
                    linewidth=1, color='blue', linestyle='--', label='y=x')

    textstr = (f"N = {n_points}\n"
               f"CC = {CC:.4f}\n"
               f"RMSE = {rmse:.4f} {unit_rmse}\n"
               # f"Bias = {bias:.4f} {unit_bias}\n"
               f"MRE = {mre_text}")
    font_stats_text = FontProperties(family='Times New Roman', size=16, weight='bold')
    current_ax.text(0.03, 0.97, textstr, transform=current_ax.transAxes,
                    fontproperties=font_stats_text, color='red',
                    verticalalignment='top', horizontalalignment='left')

    if not is_subplot_mode:
        plt.tight_layout()
        if save_path_standalone:
            plt.savefig(save_path_standalone, dpi=600, format='tiff', bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    figures_path = 'paper_figures'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    models = ['NWM', 'EF', 'UNet', 'NWM_dynamic']
    path_to_model = {'NWM': 'neuralwave', 'EF': 'earthformer', 'UNet': 'unet', 'NWM_dynamic': 'nwm_dynamic'}

    fig, axs = plt.subplots(2, 4, figsize=(24, 11), dpi=120)

    all_model_data = {}

    for col_idx, model_name in enumerate(models):
        print(f"\nProcessing model: {model_name}")

        file_path = f'results/{path_to_model[model_name]}/Integrated_Parameters_Ablation_Test/predictions_0.1.nc'

        with nc.Dataset(file_path) as f:
            print(f"Successfully opened: {file_path}")
            swh_pred_np = np.array(f.variables['SWH_Predictions'][:][:, :, 1:-1, 1:-1], dtype=np.float32)
            swh_true_np = np.array(f.variables['SWH_Labels'][:][:, :, 1:-1, 1:-1], dtype=np.float32)
            mwd_pred_np = np.array(f.variables['MWD_Predictions'][:][:, :, 1:-1, 1:-1], dtype=np.float32)
            mwd_true_np = np.array(f.variables['MWD_Labels'][:][:, :, 1:-1, 1:-1], dtype=np.float32)
            FieldLabels = torch.tensor(f.variables['FieldLabels'][:], dtype=torch.float32)

            jdg = FieldLabels.sum(dim=(-1, -2))
            mask = (jdg[:, :, 1:-1, 1:-1] > 0.01).numpy()

            swh_pred_masked = swh_pred_np[mask]
            swh_true_masked = swh_true_np[mask]
            mwd_pred_masked = mwd_pred_np[mask]
            mwd_true_masked = mwd_true_np[mask]

        plot_scatter('swh', swh_pred_masked, swh_true_masked,
                     ax=axs[0, col_idx], fig=fig, subplot_title_prefix=model_name)
        plot_scatter('mwd', mwd_pred_masked, mwd_true_masked,
                     ax=axs[1, col_idx], fig=fig, subplot_title_prefix=model_name)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    plt.savefig(f'{figures_path}/Figure_12b.tiff', dpi=300, format='tiff', bbox_inches='tight')

    # plt.show()