# -*- coding: utf-8 -*-
"""
@author: Yang
"""

import os
import matplotlib.pyplot as plt
from matplotlib import rcParams, cm
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
rcParams['lines.linewidth'] = 2.0
rcParams['lines.markersize'] = 6

def calculate_metrics(true_vals, pred_vals):
    if true_vals.size < 2:
        return np.nan, np.nan
    valid = np.isfinite(true_vals) & np.isfinite(pred_vals)
    t, p = true_vals[valid], pred_vals[valid]
    if t.size < 2:
        return np.nan, np.nan
    rmse = np.sqrt(np.mean((t - p)**2))
    mean_t = np.mean(t)
    ss_tot = np.sum((t - mean_t)**2)
    ss_res = np.sum((t - p)**2)
    r2 = 1 - ss_res/ss_tot if ss_tot != 0 else float(ss_res == 0)
    return r2, rmse

if __name__ == "__main__":
    figures_path = 'paper_figures'
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)
    models = ['NWM', 'EF', 'UNet', 'NWM(dynamic module)']
    path_to_model = {'NWM': 'neuralwave', 'EF': 'earthformer', 'UNet': 'unet', 'NWM(dynamic module)': 'nwm_dynamic'}

    idn_values = np.arange(0.1, 1.01, 0.1)

    cmaps = {
        'NWM': cm.Reds,
        'EF':  cm.Blues,
        'UNet': cm.Greens,
        'NWM(dynamic module)': cm.Greys
    }
    linestyles = {'NWM': '-', 'EF': '-', 'UNet': ':', 'NWM(dynamic module)': '--'}

    results = {
        m: {'r2': {}, 'rmse': {}} for m in models[:-1]
    }

    for model in models[:-1]:
        for idn in idn_values:
            file_path = f'results/{path_to_model[model]}/Integrated_Parameters_Ablation_Test/predictions_{idn:.1f}.nc'
            with nc.Dataset(file_path) as f:
                labels = f.variables['FieldLabels'][:]
                preds  = f.variables['FieldPredictions'][:]
            steps = labels.shape[1]
            r2_list, rmse_list = [], []
            for s in range(steps):
                true = labels[:, s, 1:-1, 1:-1].reshape(-1)
                pred = preds[:, s, 1:-1, 1:-1].reshape(-1)
                mask = true > 1e-4
                r2, rmse = calculate_metrics(true[mask], pred[mask])
                r2_list.append(r2)
                rmse_list.append(rmse)
            results[model]['r2'][idn]   = r2_list
            results[model]['rmse'][idn] = rmse_list

    file_path = 'results/nwm_dynamic/Integrated_Parameters_Ablation_Test/predictions_0.1.nc'
    with nc.Dataset(file_path) as f:
        labels = f.variables['FieldLabels'][:]
        preds  = f.variables['FieldPredictions'][:]
    steps = labels.shape[1]
    r2_dynamic, rmse_dynamic = [], []
    for s in range(steps):
        true = labels[:, s, 2:-1, 1:-1].reshape(-1)
        pred = preds[:, s, 2:-1, 1:-1].reshape(-1)
        mask = true > 1e-2
        r2, rmse = calculate_metrics(true[mask], pred[mask])
        r2_dynamic.append(r2)
        rmse_dynamic.append(rmse)

    num_steps  = len(next(iter(results[models[0]]['r2'].values())))
    steps_axis = np.arange(1, num_steps+1)

    fig, axs = plt.subplots(2, 1, figsize=(12, 8), sharex=True, dpi=300)
    for model in models[:-1]:
        cmap = cmaps[model]
        for idx, idn in enumerate(idn_values):
            t = (idn - idn_values.min()) / (idn_values.max() - idn_values.min())
            color = cmap(0.3 + 0.7 * t)
            # lw = 3.0 if model=='NWM' else 2.0
            lw = 2.0
            axs[0].plot(steps_axis, results[model]['r2'][idn],
                        color=color, linestyle=linestyles[model],
                        marker='o', markersize=4, alpha=0.8, linewidth=lw)
            axs[1].plot(steps_axis, results[model]['rmse'][idn],
                        color=color, linestyle=linestyles[model],
                        marker='o', markersize=4, alpha=0.8, linewidth=lw)

    axs[0].plot(steps_axis, r2_dynamic, marker='^', markersize=6,
                color=cmaps['NWM(dynamic module)'](0.8), linestyle='--', linewidth=2)
    axs[1].plot(steps_axis, rmse_dynamic, marker='^', markersize=6,
                color=cmaps['NWM(dynamic module)'](0.8), linestyle='--', linewidth=2)
    
    axs[0].set_ylabel('CC')
    axs[1].set_ylabel('RMSE')
    axs[1].set_xlabel('Roll-out Step')
    for ax in axs:
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.tick_params(axis='y', direction='in')
        ax.tick_params(axis='x', direction='in')
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)

    plt.xticks(np.arange(1, num_steps+1, 3))
    axs[0].set_xlim(0.5, num_steps+0.5)
    axs[1].set_xlim(0.5, num_steps+0.5)

    handles = [
        plt.Line2D([0], [0], color=cmaps[m](0.8), ls=linestyles[m], lw=3.0)
        for m in models
    ]
    labels = ['NWM', 'Earthformer', 'UNet', 'NWM(dynamic module)']
    fig.legend(handles, labels, loc='upper center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.99), fontsize=20)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(f'{figures_path}/Figure_11.tiff', dpi=300, format='tiff', bbox_inches='tight')
    # plt.show()

