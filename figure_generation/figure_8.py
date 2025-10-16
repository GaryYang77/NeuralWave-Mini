# -*- coding: utf-8 -*-
"""
@author: Yang
"""

import sys
import os
sys.path.append('..')
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import netCDF4 as nc

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']


def plot_param_compare(models, param_preds, param_trues, x, case_steps,
                       param_name, ylabel, y_limits=None, savepath=None):
    
    nrows = len(case_steps)
    ncols = len(models)
    
    x_vals = x[1:-1] 

    fig, axs = plt.subplots(nrows, ncols,
                            figsize=(5 * ncols, 4 * nrows),
                            sharex=True, sharey=False)

    if nrows == 1:
        axs = np.expand_dims(axs, axis=0)
    if ncols == 1:
        axs = np.expand_dims(axs, axis=1)

    for i, (case_idx, t_idx) in enumerate(case_steps):
        row_specific_y_limits = y_limits
        if param_name == 'MWD' and y_limits is None:
            min_val_row = np.inf
            max_val_row = -np.inf
            
            all_vals_for_row_ylim = []
            for model_in_row in models:
                if model_in_row in param_trues and model_in_row in param_preds:
                    true_vals_curr = param_trues[model_in_row][case_idx, t_idx, :, 0]
                    pred_vals_curr = param_preds[model_in_row][case_idx, t_idx, :, 0]
                    all_vals_for_row_ylim.append(true_vals_curr)
                    all_vals_for_row_ylim.append(pred_vals_curr)
            
            if all_vals_for_row_ylim:
                all_vals_for_row_ylim_flat = np.concatenate(all_vals_for_row_ylim)
                min_val_row, max_val_row = all_vals_for_row_ylim_flat.min(), all_vals_for_row_ylim_flat.max()
                
                d_y_row = max_val_row - min_val_row
                if d_y_row == 0: 
                    d_y_row = 0.2 * abs(max_val_row) if abs(max_val_row) > 1e-5 else 0.2
                    if d_y_row == 0:
                        d_y_row = 0.2
                row_specific_y_limits = (min_val_row - 0.2 * d_y_row, max_val_row + 0.2 * d_y_row)

        for j, model in enumerate(models):
            ax = axs[i, j]
            true_vals_plot = param_trues[model][case_idx, t_idx, :, 0]
            pred_vals_plot = param_preds[model][case_idx, t_idx, :, 0]

            ax.plot(x_vals, true_vals_plot, 
                    label='True' if i == 0 and j == 0 else "_nolegend_", 
                    color='blue', linewidth=2, marker='o', markersize=5)
            ax.plot(x_vals, pred_vals_plot, 
                    label='Predicted' if i == 0 and j == 0 else "_nolegend_", 
                    color='orange', linestyle='--', linewidth=2, marker='x', markersize=5)
            
            if i == 0:
                ax.set_title(model, fontsize=16, fontweight='bold')
            if j == 0:
                ax.set_ylabel(ylabel, fontsize=18, fontweight='bold')
            if i == nrows - 1:
                ax.set_xlabel('Offshore distance (m)', fontsize=18, fontweight='bold')
                xtick_values = [0, 10000, 20000, 30000, 40000, 50000]
                ax.set_xticks(xtick_values)
                ax.set_xticklabels([str(val) for val in np.flip(xtick_values)])

            ax.grid(True, linestyle='--', alpha=0.7)
            ax.tick_params(axis='both', which='major', labelsize=14)
            
            if row_specific_y_limits is not None:
                ax.set_ylim(*row_specific_y_limits)
            else: 
                all_vals_for_ylim = np.concatenate((true_vals_plot, pred_vals_plot))
                min_val, max_val = all_vals_for_ylim.min(), all_vals_for_ylim.max()
                d_y = max_val - min_val
                if d_y == 0: 
                    d_y = 0.2 * abs(max_val) if abs(max_val) > 1e-5 else 0.2
                    if d_y == 0:
                        d_y = 0.2 
                ax.set_ylim(min_val - 0.2 * d_y, max_val + 0.2 * d_y)
                
            ax.text(0.05, 0.12, 
                    f'Sample {case_idx//5 + 1}, Step {t_idx + 1}',
                    transform=ax.transAxes,
                    color='dimgray', fontsize=14,
                    verticalalignment='top')

    fig.subplots_adjust(right=0.88, hspace=0.15, wspace=0.17)

    handles, labels = axs[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels,
               loc='center right', 
               bbox_to_anchor=(1.0, 0.5), 
               fontsize=14, 
               frameon=False, columnspacing=0.5)
    
    if savepath:
        plt.savefig(savepath, dpi=300, format='tiff', bbox_inches='tight')

    # plt.show()


if __name__ == '__main__':
    figures_path = 'paper_figures'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    models = ['NWM', 'EF', 'UNet']
    path_model = {'NWM': 'neuralwave', 'EF': 'earthformer', 'UNet': 'unet'}
    idn = 0.1
    idn_str = f'{idn:.1f}'
    
    sample_indices = [6, 46]
    t_idx = 30
    sample_steps = [(si, t_idx) for si in sample_indices]

    swh_pred_all, swh_true_all = {}, {}
    mwd_pred_all, mwd_true_all = {}, {}
    x_coords = None
    for model in models:
        fpath = f'results/{path_model[model]}/Progressive_Data_Ablation_Test/predictions_{idn_str}.nc'
        print(f"Attempting to load: {fpath}")
        with nc.Dataset(fpath) as f:
            swh_pred_all[model] = np.array(
                f.variables['SWH_Predictions'][:][:, :, 1:-1, 1:-1],
                dtype=np.float32)
            swh_true_all[model] = np.array(
                f.variables['SWH_Labels'][:][:, :, 1:-1, 1:-1],
                dtype=np.float32)
            mwd_pred_all[model] = np.array(
                f.variables['MWD_Predictions'][:][:, :, 1:-1, 1:-1],
                dtype=np.float32)
            mwd_true_all[model] = np.array(
                f.variables['MWD_Labels'][:][:, :, 1:-1, 1:-1],
                dtype=np.float32)
            
            x_coords = np.array(f.variables['x'][:], dtype=np.float32)

    active_models = list(swh_pred_all.keys())

    plot_param_compare(active_models, swh_pred_all, swh_true_all, x_coords,
                       sample_steps, 'SWH', 'SWH (m)', 
                       y_limits=(0.5, 1.0),
                       savepath=f'{figures_path}/Figure_8a.tiff')
    plot_param_compare(active_models, mwd_pred_all, mwd_true_all, x_coords,
                       sample_steps, 'MWD', 'MWD (°)',
                       y_limits=None,
                       savepath=f'{figures_path}/Figure_8b.tiff')