# inference/config.py

import os
import torch
import numpy as np

# --- Basic Setup ---
SEED = 42
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32

# --- Path Definitions ---
# 假设此脚本在 nwm_mini_review/inference/ 目录下运行
# 使用 os.path.abspath 和 '..' 来确保路径的鲁棒性
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'datasets/LeadTime_TrainDataset/ww3_tp14casesRandom_leadtime5_wind.nc')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'datasets/LeadTime_TestDataset/ww3_tp14casesRandom_leadtime34_wind.nc')
MODEL_DIR = os.path.join(BASE_DIR, 'welltrained_case_models')
NOISE_DIR = os.path.join(BASE_DIR, 'datasets/noise_injections')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
NWM_CONFIG_PATH = os.path.join(BASE_DIR, 'models/nwm/default_config.yaml')
EF_CONFIG_PATH = os.path.join(BASE_DIR, 'models/earthformer/cfg_default.yaml')

# --- Test Case Definitions ---
TEST_CASES = {
    'spec_ablation': {
        'name': 'Progressive_Data_Ablation_Test',
        'model_prefix': 'model'
    },
    'intparam_ablation': {
        'name': 'Integrated_Parameters_Ablation_Test',
        'model_prefix': 'model_p'
    },
    'noise_robustness': {
        'name': 'Noise_Injected_Robustness_Test',
        'model_prefix': 'model_noise'
    }
}