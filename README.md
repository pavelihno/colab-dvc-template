# Google Colab + DVC + MLflow template

A machine learning project template with MLflow experiment tracking and DVC version control.

## Project Structure

```
├── configs/         # MLflow configuration files
├── data/            # Data directory (tracked with DVC)
├── models/          # Saved models
├── notebooks/       # Jupyter notebooks for experiments
│   └── colab.ipynb
├── src/            # Source code
│   ├── train.py    # Training script
│   ├── eval.py     # Evaluation script
│   └── model/      # Model definitions
│       └── sample.py
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Initialize MLflow

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser.

## Usage

### Configuration Files

The project uses YAML configuration files in the `configs/` directory to manage all training and evaluation parameters. This makes it easy to:

-   Track experiment settings
-   Reproduce results
-   Switch between different configurations
-   Version control your experiments

### Training a Model

Train using a specific config file:

```bash
python src/train.py --config configs/default.yaml
```

The training script will output a **run_id** at the end, which you'll need for evaluation. Example output:

```
Model logged to MLflow with run_id: abc123def456...
```

### Evaluating a Model

Evaluate a trained model using its MLflow run_id (obtained from training):

```bash
python src/eval.py --config configs/default.yaml --run_id <RUN_ID_FROM_TRAINING>
```

### Viewing Results in MLflow

Start the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser to:

-   Select your experiment
-   View and compare different runs with their parameters and metrics
-   Visualize training curves and download models

## Using in Google Colab

Open `notebooks/colab.ipynb` in Google Colab for a complete workflow:

### Workflow Overview

1. **Setup**: Install dependencies and clone repository
2. **Configure DVC**: Set up Google Drive authentication with service account
3. **Pull existing data**: Download any existing data/models from Google Drive
4. **Train model**: Run training - MLflow tracks experiments locally in `./mlruns`
5. **Evaluate model**: Test the trained model
6. **Sync to cloud**: Push data and model artifacts to Google Drive using DVC

## Data Version Control (DVC)

This template uses DVC to version control files and sync them with Google Drive. DVC tracks:

-   **Training/test data** (`data/` directory)
-   **Model weights** (`models/` directory)

### Setting up DVC with Google Drive

#### 1. Initialize DVC (if not already done)

```bash
dvc init
```

#### 2. Get Google Drive Folder ID

1. Create a folder in your Google Drive for DVC storage
2. Open the folder in your browser
3. Copy the folder ID from the URL: `https://drive.google.com/drive/folders/<FOLDER_ID>`

#### 3. Configure DVC Remote

```bash
# Add Google Drive as remote storage
dvc remote add -d gdrive gdrive://<FOLDER_ID>

# Configure DVC to use Google Drive
dvc remote modify gdrive gdrive_acknowledge_abuse true
```

Edit `.dvc/config` to verify it looks like:

```ini
[core]
    remote = gdrive
['remote "gdrive"']
    url = gdrive://<FOLDER_ID>
    gdrive_acknowledge_abuse = true
```

#### 4. Track Data and Models with DVC

```bash
# Add data files to DVC tracking
dvc add data/X_train.npy data/y_train.npy data/X_test.npy data/y_test.npy

# Add model weights to DVC tracking
dvc add models/default.keras

# Commit DVC files to Git
git add data/*.dvc data/.gitignore models/*.dvc models/.gitignore
git commit -m "Track data and models with DVC"

# Push to Google Drive
dvc push
```
