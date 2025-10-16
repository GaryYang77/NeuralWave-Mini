# -*- coding: utf-8 -*-
"""
@author: Yang
"""

import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np
import netCDF4 as nc

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['font.size'] = 14
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 14
rcParams['axes.linewidth'] = 1.5
rcParams['xtick.major.width'] = 1.5
rcParams['ytick.major.width'] = 1.5
rcParams['xtick.minor.width'] = 1.0
rcParams['ytick.minor.width'] = 1.0
rcParams['lines.linewidth'] = 2.0
rcParams['lines.markersize'] = 6

def calculate_metrics(true_vals, pred_vals):
    if true_vals.size < 2:
        return np.nan, np.nan
    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
    true_vals, pred_vals = true_vals[valid], pred_vals[valid]
    if true_vals.size < 2:
        return np.nan, np.nan
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    mean_true = np.mean(true_vals)
    ss_tot = np.sum((true_vals - mean_true) ** 2)
    ss_res = np.sum((true_vals - pred_vals) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float(ss_res == 0)

    return r2, rmse

if __name__ == "__main__":
    figures_path = 'paper_figures'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    models = ['NWM', 'EF', 'UNet', 'NWM(dynamic module)']
    path_model = {'NWM': 'neuralwave', 'EF': 'earthformer', 'UNet': 'unet', 'NWM(dynamic module)': 'nwm_dynamic'}
    linestyles = {'NWM': '-', 'EF': '-', 'UNet': ':', 'NWM(dynamic module)': '--'}
    colors = {'NWM': 'red', 'EF': 'blue', 'UNet': 'green', 'NWM(dynamic module)': 'grey'}
    results = {m: {'r2': [], 'rmse': []} for m in models}
    spectral_energy_threshold = 1e-2
    
    idn = 0.1

    num_steps = None
    for model in models[:-1]:
        file_path = f'results/{path_model[model]}/Progressive_Data_Ablation_Test/predictions_{idn:.1f}.nc'
        with nc.Dataset(file_path) as f:
            labels = f.variables['FieldLabels'][:]
            preds = f.variables['FieldPredictions'][:]
        steps = labels.shape[1]
        if num_steps is None:
            num_steps = steps
        for s in range(num_steps):
            if s >= steps:
                results[model]['r2'].append(np.nan)
                results[model]['rmse'].append(np.nan)
                continue
            true = labels[:, s, 2:-1, 1:-1].reshape(-1)
            pred = preds[:, s, 2:-1, 1:-1].reshape(-1)
            mask = true > spectral_energy_threshold
            r2, rmse = calculate_metrics(true[mask], pred[mask])
            results[model]['r2'].append(r2)
            results[model]['rmse'].append(rmse)

    file_path = 'results/nwm_dynamic/Progressive_Data_Ablation_Test/predictions_0.1.nc'
    with nc.Dataset(file_path) as f:
        labels = f.variables['FieldLabels'][:]
        preds  = f.variables['FieldPredictions'][:]
    steps = labels.shape[1]
    r2_dynamic, rmse_dynamic = [], []
    for s in range(steps):
        true = labels[:, s, 2:-1, 1:-1].reshape(-1)
        pred = preds[:, s, 2:-1, 1:-1].reshape(-1)
        mask = true > spectral_energy_threshold
        r2, rmse = calculate_metrics(true[mask], pred[mask])
        r2_dynamic.append(r2)
        rmse_dynamic.append(rmse)
        
    results['NWM(dynamic module)']['r2'] = r2_dynamic
    results['NWM(dynamic module)']['rmse'] = rmse_dynamic

    steps_axis = np.arange(1, num_steps + 1)
    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=300)

    for model in models:
        linewidth = 3.0 if model == 'NWM' else 2.0
        marker = 'o' if model != 'NWM(dynamic module)' else '^'
        marker_size = 4 if model != 'NWM(dynamic module)' else 8
        axs[0].plot(steps_axis, results[model]['r2'], label=model if model != "EF" else "Earthformer",
                    color=colors[model], linestyle=linestyles[model], marker=marker, markersize=marker_size, alpha=0.8, linewidth=linewidth)
        axs[1].plot(steps_axis, results[model]['rmse'],
                    color=colors[model], linestyle=linestyles[model], marker=marker, markersize=marker_size, alpha=0.8, linewidth=linewidth)

    axs[0].set_ylabel('CC')
    axs[1].set_ylabel('RMSE')
    axs[1].set_xlabel('Roll-out Step')
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', which='major', direction='in')
        ax.tick_params(axis='x', which='major', direction='in')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    major_ticks_step = 3
    x_ticks_major = np.arange(1, num_steps + 1, major_ticks_step)
    plt.xticks(x_ticks_major)
    axs[0].set_xlim(0.5, num_steps + 0.5)
    axs[1].set_xlim(0.5, num_steps + 0.5)
    
    handles, labels = axs[0].get_legend_handles_labels()
    labels = ['NWM', 'Earthformer', 'UNet', 'NWM(dynamic module)']
    fig.legend(handles, labels, loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{figures_path}/Figure_6.tiff', dpi=300, format='tiff', bbox_inches='tight')
    # plt.show()