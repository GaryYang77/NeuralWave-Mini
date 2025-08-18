import os
import sys
import torch
import numpy as np
import xarray as xr
import random
import argparse
from sklearn.metrics import r2_score
import scipy.io as sio

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
if os.path.basename(BASE_DIR) == 'inference':
    PARENT_DIR = os.path.join(BASE_DIR, '..')
    if PARENT_DIR not in sys.path:
        sys.path.append(PARENT_DIR)

from inference.handlers import NWMHandler, EFHandler, UNetHandler
from inference import configs
from models.nwm import get_wave_intparams

def set_seed(seed):
    """Sets the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_metrics(preds_spec, labels_spec, preds_swh, labels_swh, preds_mwd, labels_mwd):
    """Calculates and returns RMSE and R2 for spec, swh, and mwd."""
    metrics = {}

    preds_spec = np.concatenate(preds_spec[:,None], axis=0)
    labels_spec = np.concatenate(labels_spec[:,None], axis=0)

    # Spectra metrics (use a small epsilon to avoid division by zero in R2)
    rmse_spec = np.sqrt(np.mean((preds_spec - labels_spec) ** 2))
    r2_spec = r2_score(labels_spec, preds_spec)
    metrics['spec'] = {'rmse': rmse_spec, 'r2': r2_spec}
    
    # SWH metrics
    rmse_swh = np.sqrt(np.mean((preds_swh - labels_swh) ** 2))
    r2_swh = r2_score(labels_swh, preds_swh)
    metrics['swh'] = {'rmse': rmse_swh, 'r2': r2_swh}

    # MWD metrics
    rmse_mwd = np.sqrt(np.mean((preds_mwd - labels_mwd) ** 2))
    r2_mwd = r2_score(labels_mwd, preds_mwd)
    metrics['mwd'] = {'rmse': rmse_mwd, 'r2': r2_mwd}
    
    return metrics

def save_predictions_to_netcdf(save_path, preds_spec, labels_spec, preds_swh, labels_swh, preds_mwd, labels_mwd, handler):
    """Saves all predictions and labels to a NetCDF file."""
    # Ensure the save directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    ds_pred = xr.Dataset(
        {
            'FieldLabels': (['sample', 'time', 'height', 'width', 'dir', 'freq'], labels_spec),
            'FieldPredictions': (['sample', 'time', 'height', 'width', 'dir', 'freq'], preds_spec),
            'SWH_Labels': (['sample', 'time', 'height', 'width'], labels_swh),
            'SWH_Predictions': (['sample', 'time', 'height', 'width'], preds_swh),
            'MWD_Labels': (['sample', 'time', 'height', 'width'], labels_mwd),
            'MWD_Predictions': (['sample', 'time', 'height', 'width'], preds_mwd),
            'depths': (['sample', 'height', 'width'], handler.test_data_raw['dpt'].numpy())
        },
        coords={
            'sample': np.arange(labels_spec.shape[0]),
            'time': np.arange(labels_spec.shape[1]),
            'height': np.arange(labels_spec.shape[2]),
            'width': np.arange(labels_spec.shape[3]),
            'freq': handler.test_data_raw['frequency'].numpy(),
            'dir': handler.test_data_raw['direction_rad'].numpy(),
            'x': handler.test_data_raw['x'],
            'y': handler.test_data_raw['y'],
            'frequency': handler.test_data_raw['frequency'].numpy(),
            'direction': handler.test_data_raw['direction_deg'].numpy()
        }
    )
    ds_pred.to_netcdf(save_path)
    print(f"Predictions saved to {save_path}")

def run_test(model_name, test_case_key):
    """Main function to run the testing pipeline."""
    set_seed(configs.SEED)
    print(f"Using device: {configs.DEVICE}")

    handler_map = {
        "neuralwave": NWMHandler,
        "earthformer": EFHandler,
        "unet": UNetHandler,
        "nwm_dynamic": NWMHandler
    }
    
    if model_name not in handler_map:
        raise ValueError(f"Model '{model_name}' not supported. Choose from {list(handler_map.keys())}")
    
    if model_name == 'nwm_dynamic':
        handler = handler_map[model_name](configs.DEVICE, learnstot=False)
    else:
        handler = handler_map[model_name](configs.DEVICE)

    case_info = configs.TEST_CASES[test_case_key]
    save_dir = os.path.join(configs.RESULTS_DIR, model_name, case_info['name'])
    
    # Initialize results dictionary to collect all metrics
    results_dict = {
        'id': [],
        'rmse_spec': [],
        'r2_spec': [],
        'rmse_swh': [],
        'r2_swh': [],
        'rmse_mwd': [],
        'r2_mwd': []
    }

    for param in np.arange(0.1, 1.1, 0.1):
        param_str = "{:.1f}".format(param)
        print(f"\n--- Running Test ---")
        print(f"Model: {model_name}, Case: {case_info['name']}, Parameter: {param_str}")

        # 1. Get DataLoader for the specific run
        data_loader = handler.get_dataloader(test_case_key, param)
        
        # 2. Load the corresponding model
        model_path = handler.get_model_path(model_name, test_case_key, param)
        if not os.path.exists(model_path):
            print(f"Warning: Model path not found, skipping: {model_path}")
            continue
        model = handler.load_model(model_path)

        # 3. Get predictions
        preds_spec_np = handler.predict(data_loader)
        labels_spec_np = handler.test_data_raw['FieldLabels_test'].numpy()

        # 4. Calculate integrated parameters (SWH, MWD)
        print("Calculating integrated parameters (SWH, MWD)...")
        direction_deg = handler.test_data_raw['direction_deg']
        frequency = handler.test_data_raw['frequency']

        preds_swh, preds_mwd = get_wave_intparams(torch.from_numpy(preds_spec_np), theta=direction_deg, freq=frequency)
        labels_swh, labels_mwd = get_wave_intparams(torch.from_numpy(labels_spec_np), theta=direction_deg, freq=frequency)
        
        preds_swh, preds_mwd = preds_swh.numpy(), preds_mwd.numpy()
        labels_swh, labels_mwd = labels_swh.numpy(), labels_mwd.numpy()

        # 5. Calculate and print metrics
        mk_spec_test = np.zeros_like(preds_spec_np, dtype=bool)
        mk_intparams_test = np.zeros_like(preds_swh, dtype=bool)
        mk_spec_test[:,:,1:-1,1:-1] = True
        mk_intparams_test[:,:,1:-1,1:-1] = True
        
        jdg = labels_spec_np.sum(axis=(-1,-2))
        valid_mask_swh_mwd = (jdg[:, :, 1:-1, 1:-1] - 1e-2) >= 0  # mask
        valid_mask_spec = labels_spec_np[:,:,1:-1,1:-1].sum(axis=(0,1,2,3)) > 1e-2  # mk_spec
        valid_mask_spec = np.broadcast_to(valid_mask_spec, mk_spec_test[:,:,1:-1,1:-1].shape)
        spec_mask = mk_spec_test[:, :, 1:-1, 1:-1, ...] & valid_mask_spec
        intparams_mask = mk_intparams_test[:, :, 1:-1, 1:-1] & valid_mask_swh_mwd
        
        metrics = calculate_metrics(
            preds_spec_np[:, :, 1:-1, 1:-1, ...][spec_mask],
            labels_spec_np[:, :, 1:-1, 1:-1, ...][spec_mask],
            preds_swh[:, :, 1:-1, 1:-1][intparams_mask],
            labels_swh[:, :, 1:-1, 1:-1][intparams_mask],
            preds_mwd[:, :, 1:-1, 1:-1][intparams_mask],
            labels_mwd[:, :, 1:-1, 1:-1][intparams_mask]
        )
        
        print(f"Results for param {param_str}:")
        print(f"  Spectra | RMSE: {metrics['spec']['rmse']:.6f}, R²: {metrics['spec']['r2']:.6f}")
        print(f"  SWH     | RMSE: {metrics['swh']['rmse']:.6f}, R²: {metrics['swh']['r2']:.6f}")
        print(f"  MWD     | RMSE: {metrics['mwd']['rmse']:.6f}, R²: {metrics['mwd']['r2']:.6f}")

        results_dict['id'].append(param)
        results_dict['rmse_spec'].append(metrics['spec']['rmse'])
        results_dict['r2_spec'].append(metrics['spec']['r2'])
        results_dict['rmse_swh'].append(metrics['swh']['rmse'])
        results_dict['r2_swh'].append(metrics['swh']['r2'])
        results_dict['rmse_mwd'].append(metrics['mwd']['rmse'])
        results_dict['r2_mwd'].append(metrics['mwd']['r2'])

        # 6. Save results to NetCDF
        save_path = os.path.join(save_dir, f'predictions_{param_str}.nc')
        save_predictions_to_netcdf(save_path, preds_spec_np, labels_spec_np, preds_swh, labels_swh, preds_mwd, labels_mwd, handler)
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sio.savemat(os.path.join(save_dir, 'results.mat'), results_dict)
    print(f"Results summary saved to {os.path.join(save_dir, 'results.mat')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run inference for NWM, EF, or UNet models.")
    parser.add_argument('--model', type=str, required=True, choices=['neuralwave', 'earthformer', 'unet','nwm_dynamic'],
                        default='neuralwave',
                        help='Name of the model to test.')
    parser.add_argument('--case', type=str, required=True, choices=configs.TEST_CASES.keys(),
                        help='The test case to run.')
    
    args = parser.parse_args()

    run_test(args.model, args.case)