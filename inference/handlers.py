# inference/handlers.py
import os
import torch
import xarray as xr
import numpy as np
from torch.utils.data import DataLoader, Dataset
from abc import ABC, abstractmethod
import sys
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from models.nwm import NeuralWave, get_wave_intparams
from models.earthformer import CuboidWaveModel
from models.unet import UNet
from inference import configs

# --- Helper Functions (from original scripts) ---

def get_random_mask(depth, random_choice, multi_mask=False):
    np.random.seed(configs.SEED)
    if multi_mask:
        mask = np.random.choice(random_choice, size=depth.shape[0:3])
    else:
        mask = np.random.choice(random_choice, size=depth.shape[1:3])
        mask = np.tile(mask, (depth.shape[0], 1, 1))
    return torch.tensor(mask, dtype=torch.int32)

# --- Base Handler Class ---

class BaseModelHandler(ABC):
    """Abstract Base Class for model handlers."""
    def __init__(self, device):
        self.device = device
        self.model = None
        self.train_stats = {}
        self.test_data_raw = {}
        self._load_common_data()

    def _load_common_data(self):
        """Loads data common to all models, like frequency and direction."""
        with xr.open_dataset(configs.TEST_DATA_PATH) as ds:
            self.test_data_raw['dpt'] = torch.tensor(ds['depths'].values, dtype=torch.float32)
            self.test_data_raw['frequency'] = torch.tensor(ds['frequency'].values, dtype=torch.float32)
            # NWM and EF/UNet use slightly different direction formats
            self.test_data_raw['direction_deg'] = torch.tensor(ds['direction'].values, dtype=torch.float32)
            self.test_data_raw['direction_rad'] = self.test_data_raw['direction_deg'] / 180 * torch.pi
            self.test_data_raw['x'] = ds['x'].values
            self.test_data_raw['y'] = ds['y'].values

    @abstractmethod
    def get_dataloader(self, test_case_name, param):
        """Prepares and returns a DataLoader for the given test case and parameter."""
        pass

    @abstractmethod
    def load_model(self, model_path):
        """Loads the model from a .pth file."""
        pass

    @abstractmethod
    def predict(self, data_loader):
        """Runs the inference loop and returns predictions."""
        pass

    def get_model_path(self, model_name, case_key, param):
        case_info = configs.TEST_CASES[case_key]
        param_str = "{:.1f}".format(param)
        model_filename = f"{case_info['model_prefix']}_{param_str}.pth"
        
        # NWM has an extra 'network' in the filename for some cases
        if model_name in ['neuralwave', 'nwm_dynamic'] and case_key in ['intparam_ablation', 'spec_ablation', 'noise_robustness']:
             model_filename = f"{case_info['model_prefix']}_network_{param_str}.pth"
             model_name = "neuralwave"

        return os.path.join(configs.MODEL_DIR, model_name + "_pths", case_info['name'].split('_Test')[0], model_filename)

# --- NWM Handler ---

class NWMHandler(BaseModelHandler):
    class NWMDataset(Dataset):
        def __init__(self, spectra, bathy, forcings, spectra_label, boundary_condition, random_mask):
            self.spectra, self.bathy, self.forcings, self.spectra_label, self.boundary_condition, self.random_mask = \
                spectra, bathy, forcings, spectra_label, boundary_condition, random_mask
        def __len__(self): return len(self.bathy)
        def __getitem__(self, idx):
            return self.spectra[idx], self.bathy[idx], self.forcings[idx], \
                   self.spectra_label[idx], self.boundary_condition[idx], self.random_mask[idx]

    def __init__(self, device, learnstot = True):
        super().__init__(device)
        self._preprocess_train_data()
        self.learnstot = learnstot

    def _preprocess_train_data(self):
        """Preprocesses training data to get normalization stats, exactly as in NWM_ModelTest.py."""
        with xr.open_dataset(configs.TRAIN_DATA_PATH) as ds:
            FieldInputs_train = torch.tensor(ds['FieldInputs'].values, dtype=torch.float32)
            self.train_stats['field_mean'] = FieldInputs_train[:, 1:-1, 1:-1].mean()
            self.train_stats['field_std'] = FieldInputs_train[:, 1:-1, 1:-1].std()
            self.train_stats['x_train'] = torch.tensor(ds['x'].values, dtype=torch.float32)
            self.train_stats['y_train'] = torch.tensor(ds['y'].values, dtype=torch.float32)
            self.train_stats['frequency_train'] = torch.tensor(ds['frequency'].values, dtype=torch.float32)
            self.train_stats['direction_train'] = torch.tensor(ds['direction'].values, dtype=torch.float32) / 180 * torch.pi

    def get_dataloader(self, test_case_name, param):
        with xr.open_dataset(configs.TEST_DATA_PATH) as ds:
            FieldInputs_test = torch.tensor(ds['FieldInputs'].values, dtype=torch.float32)
            FieldLabels_test = torch.tensor(ds['FieldLabels'].values, dtype=torch.float32)
            dpt_test = torch.tensor(ds['depths'].values, dtype=torch.float32)
            uwind_nc = torch.tensor(ds['uwind'].values, dtype=torch.float32) # uwind was mis-assigned from depths in original
            x_test = torch.tensor(ds['x'].values, dtype=torch.float32)
            direction_test = torch.tensor(ds['direction'].values, dtype=torch.float32) / 180 * torch.pi

        if test_case_name == 'noise_robustness':
            noise_path = os.path.join(configs.NOISE_DIR, f"noise_level_{param:.1f}.npy")
            noise_test = torch.tensor(np.load(noise_path), dtype=torch.float32)
            noise_test[FieldLabels_test == 0] = 0
            FieldLabels_test = torch.clamp(FieldLabels_test + noise_test, min=0)
        self.test_data_raw['FieldLabels_test'] = FieldLabels_test

        depths_test = torch.zeros_like(dpt_test); depths_test[:,1:-1,1:-1] = dpt_test[:,1:-1,1:-1]
        uwind_test = torch.zeros_like(uwind_nc); uwind_test[:, 1:-1, 1:-1] = uwind_nc[:, 1:-1, 1:-1]
        
        boundary_condition_test = torch.zeros_like(FieldInputs_test)
        boundary_condition_test[:, 1, 1] = FieldInputs_test[:, 1, 1]

        depths_emb = depths_test[:, :, 1:-1][:, None, :, :, None].repeat([1, 1, 1, 24, 1])
        uwind_emb = uwind_test[:, :, 1:-1][:, None, :, :, None].repeat([1, 1, 1, 24, 1])
        x_emb = x_test.unsqueeze(0)[:, None, :, None, None].repeat([FieldInputs_test.shape[0], 1, 1, len(direction_test), 1])
        direction_emb = direction_test.unsqueeze(0)[:, None, None, :, None].repeat([FieldInputs_test.shape[0], 1, len(x_test), 1, 1])

        field_mean, field_std = self.train_stats['field_mean'], self.train_stats['field_std']
        depths_emb = (depths_emb - depths_emb.mean()) / (depths_emb.std() + 1e-8) * field_std + field_mean
        uwind_emb = (uwind_emb - uwind_emb.mean()) / (uwind_emb.std() + 1e-8) * field_std + field_mean
        x_emb = (x_emb - x_emb.mean()) / (x_emb.std() + 1e-8) * field_std + field_mean
        direction_emb = (direction_emb - direction_emb.mean()) / (direction_emb.std() + 1e-8) * field_std + field_mean

        forcings_input = torch.cat((depths_emb, uwind_emb, x_emb, direction_emb), dim=-1)
        random_mask = get_random_mask(depths_test, [1], multi_mask=True)[:, None, :, :].repeat([1, FieldLabels_test.shape[1], 1, 1])
        
        dataset = self.NWMDataset(FieldInputs_test, depths_test, forcings_input, FieldLabels_test, boundary_condition_test, random_mask)
        return DataLoader(dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

    def load_model(self, model_path):
        self.model = NeuralWave.load_pretrained(
            model_path,
            self.train_stats['x_train'], self.train_stats['y_train'], 
            self.train_stats['direction_train'], self.train_stats['frequency_train'], 
            hypara_file=configs.NWM_CONFIG_PATH, device=self.device)
        self.model.move_to_device()
        self.model.eval()
        return self.model

    def predict(self, data_loader):
        all_spec_preds = []
        with torch.no_grad():
            for spec_ini, depth, extra_inputs, spec_label, bc, _ in data_loader:
                spec_ini, depth, extra_inputs, spec_label, bc = \
                    spec_ini.to(self.device).float(), depth.to(self.device).float(), \
                    extra_inputs.to(self.device).float(), spec_label.to(self.device).float(), \
                    bc.to(self.device).float()
                
                spec_output = torch.zeros_like(spec_label, device=self.device)
                
                spec_next = self.model(spec_ini, depth, extra_inputs, boundary_condition=bc, learnStot=self.learnstot)
                spec_output[:, 0] = spec_next
                for step in range(1, spec_label.shape[1]):
                    spec_next = self.model(spec_next, depth, extra_inputs, boundary_condition=bc, learnStot=self.learnstot)
                    spec_output[:, step] = spec_next
                
                all_spec_preds.append(spec_output.cpu())
        
        return np.concatenate([p.numpy() for p in all_spec_preds], axis=0)

# --- EF and UNet Common Handler Base ---

class SharedUnetEFHandler(BaseModelHandler):
    """Shares data preprocessing logic for UNet and Earthformer."""
    class MultiWaveDataset(Dataset):
        def __init__(self, inputs, labels):
            self.inputs, self.labels = inputs, labels
        def __len__(self): return len(self.inputs)
        def __getitem__(self, idx): return self.inputs[idx], self.labels[idx]

    def __init__(self, device):
        super().__init__(device)
        self._preprocess_train_data()
    
    def _preprocess_train_data(self):
        """Preprocesses training data to get normalization stats, as in EF/UNet scripts."""
        with xr.open_dataset(configs.TRAIN_DATA_PATH) as ds:
            FieldInputs_train = torch.tensor(ds['FieldInputs'].values, dtype=torch.float32)
            FieldInputs_reshaped = FieldInputs_train[:, 1:-1, 1:-1, :, 1:-1].permute(0, 2, 1, 3, 4)
            self.train_stats['field_mean'] = FieldInputs_reshaped.mean()
            self.train_stats['field_std'] = FieldInputs_reshaped.std()

    def get_dataloader(self, test_case_name, param):
        with xr.open_dataset(configs.TEST_DATA_PATH) as ds:
            FieldInputs_test = torch.tensor(ds['FieldInputs'].values, dtype=torch.float32)
            FieldLabels_test = torch.tensor(ds['FieldLabels'].values, dtype=torch.float32)
            depths_te = torch.tensor(ds['depths'].values, dtype=torch.float32)
            uwind_te = torch.tensor(ds['uwind'].values, dtype=torch.float32)
            x_test = torch.tensor(ds['x'].values, dtype=torch.float32).unsqueeze(0)
            direction_test = torch.tensor(ds['direction'].values, dtype=torch.float32).unsqueeze(0)

        if test_case_name == 'noise_robustness':
            noise_path = os.path.join(configs.NOISE_DIR, f"noise_level_{param:.1f}.npy")
            noise_test = torch.tensor(np.load(noise_path), dtype=torch.float32)
            noise_test[FieldLabels_test == 0] = 0
            FieldLabels_test = torch.clamp(FieldLabels_test + noise_test, min=0)
        self.test_data_raw['FieldLabels_test'] = FieldLabels_test

        FieldInputs_reshaped = FieldInputs_test[:, 1:-1, 1:-1, :, 1:-1].permute(0, 2, 1, 3, 4)
        FieldLabels_reshaped = FieldLabels_test[:, :, 1:-1, 1:-1, :, 1:-1].permute(0, 1, 2, 4, 3, 5).squeeze(-1)

        depths = depths_te[:, 1:-1, 1:-1][:, None, :, :, None].repeat([1, 1, 1, 24, 1])
        uwind = uwind_te[:, 1:-1, 1:-1][:, None, :, :, None].repeat([1, 1, 1, 24, 1])
        x_emb = x_test[:, 1:-1][:, None, :, None, None].repeat([FieldInputs_test.shape[0], 1, 1, FieldInputs_reshaped.shape[3], 1])
        direction_emb = direction_test[:, None, None, :, None].repeat([FieldInputs_test.shape[0], 1, FieldInputs_reshaped.shape[2], 1, 1])

        field_mean, field_std = self.train_stats['field_mean'], self.train_stats['field_std']
        depths = (depths - depths.mean()) / (depths.std() + 1e-8) * field_std + field_mean
        uwind = (uwind - uwind.mean()) / (uwind.std() + 1e-8) * field_std + field_mean
        x_emb = (x_emb - x_emb.mean()) / (x_emb.std() + 1e-8) * field_std + field_mean
        direction_emb = (direction_emb - direction_emb.mean()) / (direction_emb.std() + 1e-8) * field_std + field_mean
        
        input_data = torch.cat((FieldInputs_reshaped, uwind, depths, x_emb, direction_emb), dim=-1)
        dataset = self.MultiWaveDataset(input_data, FieldLabels_reshaped)
        return DataLoader(dataset, batch_size=configs.BATCH_SIZE, shuffle=False)

# --- Earthformer Handler ---

class EFHandler(SharedUnetEFHandler):
    class Earthformer(torch.nn.Module):
        def __init__(self, cfgFile):
            super().__init__()
            self.earthformer = CuboidWaveModel(total_num_steps=100, oc_file=cfgFile, save_dir='./').torch_nn_module
        def forward(self, x): return self.earthformer(x)

    def load_model(self, model_path):
        self.model = self.Earthformer(configs.EF_CONFIG_PATH).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        return self.model

    def predict(self, data_loader):
        all_spec_preds = []
        with torch.no_grad():
            for x, y_label in data_loader:
                x = x.to(self.device).float()
                preds_timestep = []
                current_input = x
                for _ in range(y_label.shape[1]):
                    delta_y = self.model(current_input)
                    y_pred = current_input[..., 0:1].clone()
                    y_pred[:, :, 1:] += delta_y[:, :, 1:]
                    y_pred.clamp_(min=0)
                    preds_timestep.append(y_pred)
                    current_input = torch.cat((y_pred, x[..., 1:]), dim=-1)
                
                preds_all_timesteps = torch.cat(preds_timestep, dim=1)
                # Add padding back to match original NWM format for saving
                spec_output = torch.zeros(y_label.shape[0], y_label.shape[1], y_label.shape[2]+2, 3, y_label.shape[3], 3, device='cpu')
                spec_output[:, :, 1:-1, 1:-1, :, 1:-1] = preds_all_timesteps.cpu()[:, :, :, None, :, :]
                all_spec_preds.append(spec_output)

        return np.concatenate([p.numpy() for p in all_spec_preds], axis=0)

# --- UNet Handler ---

class UNetHandler(SharedUnetEFHandler):
    def load_model(self, model_path):
        self.model = UNet(n_channels=5, n_classes=1).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        return self.model

    def predict(self, data_loader):
        all_spec_preds = []
        with torch.no_grad():
            for x, y_label in data_loader:
                x = x.to(self.device).float()
                # Crucial permutation for UNet
                x_unet = x.permute(0, 4, 2, 3, 1).squeeze(-1)
                
                preds_timestep = []
                current_input = x_unet
                for _ in range(y_label.shape[1]):
                    delta_y = self.model(current_input)
                    y_pred = current_input[:, 0:1, :, :].clone()
                    y_pred[:, :, 1:] += delta_y[:, :, 1:]
                    y_pred.clamp_(min=0)
                    preds_timestep.append(y_pred)
                    current_input = torch.cat((y_pred, x_unet[:, 1:, :, :]), dim=1)
                
                preds_all_timesteps = torch.cat(preds_timestep, dim=1)
                # Add padding back to match original NWM format for saving
                spec_output = torch.zeros(y_label.shape[0], y_label.shape[1], y_label.shape[2]+2, 3, y_label.shape[3], 3, device='cpu')
                spec_output[:, :, 1:-1, 1:-1, :, 1:-1] = preds_all_timesteps.cpu()[:, :, :, None, :, None]
                all_spec_preds.append(spec_output)
        
        return np.concatenate([p.numpy() for p in all_spec_preds], axis=0)