# NeuralWave Mini: A Hybrid Ocean Wave Modeling Framework

[![Dataset & Models (Zenodo)](https://img.shields.io/badge/Zenodo-Data%26Models-blue)](<https://doi.org/10.5281/zenodo.16889873>)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org)

## Overview

NeuralWave Mini is the official **proof-of-concept (POC)** implementation of our hybrid ocean wave modeling framework, as presented in our manuscript. It is a lightweight and focused version designed specifically to reproduce the **one-dimensional (1D) wave propagation experiments (including depth refraction)** that form the basis of our initial study.

The primary goals of this repository are:

- **Research Reproducibility**: To provide a direct and verifiable way to reproduce all 1D experimental results presented in our paper.
- **Educational Starting Point**: To offer a simplified and accessible 1D model, ideal for researchers and students looking to understand the core principles of our hybrid modeling approach without the complexity of a full 2D system.
- **Foundation for Future Work**: To provide a foundational codebase for the community to build upon while our full-scale, 2D research model is under development.

### Scope of this 'Mini' Version

⚠️ **Please Note**: This implementation has a specific and limited scope:

- **1D Simulation Only**: The model is strictly **one-dimensional (1D)**, mirroring the focus of our initial proof-of-concept paper.
- **Designed for Reproducibility**: This code is optimized for reproducing our published results, not for operational forecasting or general-purpose 2D simulations.
- **Future Development**: The full-scale NeuralWave framework, capable of handling complex two-dimensional scenarios, is an ongoing research project and will be released separately in the future.

## Architecture

Our hybrid framework integrates multiple state-of-the-art models:

- **NeuralWave**: Our hybrid wave model that combines ocean wave dynamics framework with U-Net neural network components
- **EarthFormer**: Transformer-based spatiotemporal modeling
- **U-Net**: Convolutional neural network for spatial feature extraction

## Project Structure

```
nwm_mini_openResearch_Final/
├── datasets/                    # Training and test datasets (download required)
├── welltrained_case_models/     # Pre-trained model weights (download required)
├── results/                     # Experimental results (download required)
├── paper_figures/               # Generated figures from the paper (download required)
├── models/                      # Model implementations
│   ├── nwm/                     # NeuralWave model
│   ├── earthformer/             # EarthFormer implementation
│   └── unet/                    # U-Net implementation
├── figure_generation/           # Scripts for generating paper figures
├── inference/                   # Inference handlers and configurations
├── run_inference.py             # Main inference script
└── requirements.txt             # Python dependencies
```

## Quick Start

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/GaryYang77/NeuralWave-Mini.git
cd NeuralWave-Mini
```

2. **Create and activate virtual environment**
```bash
python -m venv nwm_env
# Windows
nwm_env\Scripts\activate
# Linux/Mac
source nwm_env/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Download required data and models**

⚠️ **Important**: The datasets and pre-trained models are hosted on Zenodo and must be downloaded separately. The required files are split into two parts due to their size.

*   **[Datasets, Pre-trained models & Results](<https://doi.org/10.5281/zenodo.16889873>)**

After downloading, extract **both** archives and place the contained folders (`datasets/`, `welltrained_case_models/`, `results/`, `paper_figures/`) into the root of this project directory to match the structure shown above.

## Usage

### Running Inference

The main inference script supports multiple models and experimental cases:

```bash
# Run NeuralWave model with spectra ablation case
python run_inference.py --model neuralwave --case spec_ablation

# Run EarthFormer model
python run_inference.py --model earthformer --case progressive_ablation

# Run U-Net model
python run_inference.py --model unet --case noise_robustness
```

### Generating Paper Figures

Reproduce the exact figures from our paper:

```bash
# Generate specific figures
python figure_generation/figure_4a.py    # Spectra ablation analysis
python figure_generation/figure_10.py    # Integrate wave parameters ablation analysis
python figure_generation/figure_12a.py   # Noise robustness analysis
```

**Generated figures will be saved in `paper_figures/` directory**

## Experimental Cases

Our framework supports three main experimental categories:

### 1. Progressive Data Ablation Test
Evaluates model performance with varying amounts of training data.
```bash
python run_inference.py --model neuralwave --case progressive_ablation
```

### 2. Integrated Parameters Ablation Test
Analyzes the impact of different integrated wave parameters.
```bash
python run_inference.py --model neuralwave --case integrated_params
```

### 3. Noise Injection Robustness Test
Tests model robustness against various noise levels (0.1-1.0).
```bash
python run_inference.py --model neuralwave --case noise_robustness
```

## Configuration

Model configurations are stored in:
- `experiments/cfg.yaml` - EarthFormer configuration
- `models/nwm/default_config.yaml` - NeuralWave configuration
- `inference/configs.py` - General inference settings

## Citation

This code supports our manuscript, which is currently under review. If you use this code or our methodology in your research, we kindly ask you to cite our paper.

**Manuscript Reference:**
> Yang, G. G., Lu, W., et al. (2025). "Unveiling the Potential of End-to-End Ocean Wave Modeling via a Hybrid Framework." *Submitted to Journal of Advances in Modeling Earth Systems (JAMES)*.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or discussions, please use the appropriate channel:

- **Technical Issues & Code Questions**: For bug reports, installation problems, or questions about the code, please **open an issue** on the [GitHub Issues](https://github.com/GaryYang77/NeuralWave-Mini/issues) page. This is the preferred method for technical support.

- **Primary Code Author**: For direct inquiries not suitable for a public issue, you can reach out to:
  - **GaryYang**: `yanggy25@mail2.sysu.edu.cn`

## Version History

- **v1.0.0** (2025-08-19): Initial mini version release

---

**Note**: This is the mini version of our NeuralWave framework. The complete version with additional features and optimizations will be released upon completion of our ongoing research.
