# NRAD OpenData Pipeline

## Overview

This repository implements a complete high-energy physics analysis pipeline for ATLAS Open Data, focused on Classification Without Labels (CWoLa) anomaly detection. It combines ATLAS event streaming, dataset creation, data preparation, ML-based background modeling, CWoLa signal-region studies, and anomaly score evaluation.

The analysis is organized into numbered steps from raw ROOT data/MC streaming through final anomaly detection. It uses ATLAS metadata, ROOT/ROOT RDataFrame reductions, parquet datasets, PyTorch models, and CWoLa-style background classification to detect anomalous jet events.

## Pipeline Architecture

### Step 0: Data / MC Streaming

- `0_NRAD_STREAM_DATA.py`
  - Streams ATLAS data by physics run period.
  - Reads ATLAS metadata JSON and selects ROOT files for each run.
  - Reduces full xAOD ROOT branch content to a smaller set of relevant branches.
  - Writes output into `NRAD_Dataset/Data/reduce_root` (not in this repository). [About 3.3 TB]

- `0_NRAD_STREAM_MC.py`
  - Streams ATLAS MC samples for processes such as `Wjets`, `Zjets`, `ttbar`, `Single_top`, `Multijet`, and `Diboson`.
  - Reads MC metadata JSON files like `ATLAS_boson.json`, `ATLAS_ttbar.json`, `ATLAS_QCD.json`.
  - Extracts and reduces MC branches for the same analysis features used in data.
  - Writes output into `NRAD_Dataset/MC/reduce_root` (not in this repository). [About 842 GB]

### Step 1: Dataset Creation

- `1_NRAD_Dataset.ipynb`
  - Builds the first analysis datasets from reduced ROOT files to `Regions_data` and `Regions_MC`.
  - Loads reduced ROOT content with `uproot`, `awkward`, `polars`, and `pandas`.
  - Applies event selection, feature extraction, and dataset formatting.
  - Produces structured parquet / dataset files ready for ML.

### Step 2: Data Preparation

- `2_NRAD_PREP.ipynb`
  - Prepares the created datasets for training and evaluation.
  - Applies feature scaling and region selection.
  - Splits datasets into control regions (CR) and signal region (SR) as needed.
  - Creates final dataset folders such as `Final_Dataset_ATLAS/`, `Final_Dataset_ML/`, and region-specific parquet files.

### Step 3: Training & Reweighting

- `3_NRAD_TRAIN_Generate.ipynb`
  - Trains generative models for background feature generation.
  - Uses `SimpleMAF` normalizing flows to learn data-like feature distributions in CRs.
  - Produces generative background models of the features using the context variables.

- `3_NRAD_TRAIN_Reweight.ipynb`
  - Trains reweighting models to improve MC/data agreement.
  - Uses classifiers and transfer factors to correct MC weights.
  - Generates reweighted MC and updated background predictions.

### Step 4: CWoLa Analysis

- `4_NRAD_CWOLA_SR.ipynb`
  - Builds the CWoLa signal-region analysis.
  - Trains classifiers using signal-region data and background-like samples.
  - Evaluates classifier scores, extrapolation, and closure.

- `4_NRAD_CWOLA_CR.ipynb`
  - Performs CWoLa analysis in control regions.
  - Produces validation plots, MC/data comparisons, and checks for model bias.
  - Evaluation directories: `Eval_CWoLa_SR_*`, `Eval_CWoLa_CR_*`

### Step 5: Anomaly Detection

- `5_NRAD_Anomaly.ipynb`
  - Uses trained CWoLa models and reweighted backgrounds to search for anomalies.
  - Calculates anomaly scores, signal significance, and score distributions.

## Repository Structure

- `0_NRAD_STREAM_DATA.py` - Data streaming and ROOT reduction for ATLAS data.
- `0_NRAD_STREAM_MC.py` - MC streaming and ROOT reduction for ATLAS MC samples.
- `1_NRAD_Dataset.ipynb` - Dataset creation notebook.
- `2_NRAD_PREP.ipynb` - Data preparation notebook.
- `3_NRAD_TRAIN_Generate.ipynb` - Generative model training notebook.
- `3_NRAD_TRAIN_Reweight.ipynb` - Reweighting/model training notebook.
- `4_NRAD_CWOLA_SR.ipynb` - CWoLa signal region analysis notebook.
- `4_NRAD_CWOLA_CR.ipynb` - CWoLa control region analysis notebook.
- `5_NRAD_Anomaly.ipynb` - Final anomaly detection notebook.

Supporting directories:

- `configs/` - YAML configuration files for training and model hyperparameters.
- `Regions_data`, `Regions_MC` - Dataset with prepared Event Selections
- `Final_Dataset_ATLAS/`, `Final_Dataset_ML/` - Prepared datasets for ATLAS and ML workflows.
- `Eval_CWoLa_*` - Evaluation and model outputs and plots for CWoLa studies.
- `Models_Extrapolation_ATLAS/`, `Models_Extrapolation_ML/` - CWoLa trained model outputs.
- `model_scripts/` - Model definitions and training utilities.
- `ATLAS_*.json` - ATLAS metadata files used for dataset streaming.

## Prerequisites & Dependencies

The pipeline requires a HEP analysis environment with ROOT and Python packages for data processing, ML training, and plotting.

Recommended software:

- Python 3.8+ (or compatible Python 3.x)
- ROOT with PyROOT support

Key Python packages:

- `numpy`
- `pandas`
- `polars`
- `uproot`
- `awkward`
- `scikit-learn`
- `torch`
- `matplotlib`
- `scipy`
- `joblib`
- `pyyaml`
- `root_numpy` / `uproot` + `ROOT` for PyROOT access

Note: Some notebooks use hard-coded local paths such as `/home/aegis/...`. Update these paths to match your environment before running.

## How to Run

The pipeline is intended to be executed in order from step 0 through step 5.

1. Step 0: Stream and reduce input data/MC
   - `python 0_NRAD_STREAM_DATA.py -period <PERIOD>`
   - `python 0_NRAD_STREAM_MC.py -process <PROCESS>`
   - Example processes: `Wjets`, `Zjets`, `ttbar`, `Single_top`, `Multijet`, `Diboson`

2. Step 1: Create analysis datasets
   - Open and run `1_NRAD_Dataset.ipynb`
   - This notebook produces structured datasets from the reduced ROOT outputs.

3. Step 2: Prepare final ML datasets
   - Open and run `2_NRAD_PREP.ipynb`
   - Use this notebook to generate region splits, scaling, and final training/testing parquet files.

4. Step 3: Train models and reweight MC
   - Open and run `3_NRAD_TRAIN_Generate.ipynb` to train generative models.
   - Open and run `3_NRAD_TRAIN_Reweight.ipynb` to train MC reweighting/classification models.
   - Alternatively, use helper scripts in `train_generate.py`, `train_reweight.py`, and `train_contextW.py`.

5. Step 4: Run CWoLa analysis
   - Open and run `4_NRAD_CWOLA_SR.ipynb` for signal region CWoLa studies.
   - Open and run `4_NRAD_CWOLA_CR.ipynb` for control region validation and closure tests.

6. Step 5: Perform anomaly detection
   - Open and run `5_NRAD_Anomaly.ipynb`
   - This notebook evaluates anomaly scores and can compare against signal benchmarks.

## Notes

- Many notebooks and scripts expect files and directories created by earlier pipeline steps. Run steps sequentially to avoid missing inputs.
- Some workflows rely on locally configured environment paths. Search for `/home/aegis` in notebooks and adjust to your setup.
- The repository contains both development notebooks and production-like helper scripts, so use the best fit for your environment.

## Contact

For questions about the NRAD OpenData pipeline or to adapt it for a different ATLAS dataset, review the notebook code comments and configuration files in `configs/`.
