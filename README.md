# colab-dvc-mlflow-template

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

MLflow will automatically create a local tracking directory in `./mlruns`.

To view the MLflow UI locally:

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

**Note:** The trained model is saved both locally (in `models/` directory) and logged to MLflow (in `mlruns/` directory) for experiment tracking and easy retrieval.

### Evaluating a Model

Evaluate a trained model using its MLflow run_id (obtained from training):

```bash
python src/eval.py --config configs/default.yaml --run_id <RUN_ID_FROM_TRAINING>
```

**Note:** The evaluation script loads the model directly from MLflow using the run_id, ensuring you're evaluating the exact model version that was logged during training.

### Viewing Results in MLflow

Start the MLflow UI:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser to:

- Select your experiment
- View and compare different runs with their parameters and metrics
- Visualize training curves and download models

## Using in Google Colab

Open `notebooks/colab.ipynb` in Google Colab for a complete workflow:

### Workflow Overview

1. **Setup**: Install dependencies and clone repository
2. **Configure DVC**: Set up Google Drive authentication with service account
3. **Pull existing data**: Download any existing data/models/experiments from Google Drive
4. **Train model**: Run training - MLflow tracks experiments locally in `./mlruns`
5. **Evaluate model**: Test the trained model
6. **Sync to cloud**: Push all artifacts (data, models, MLflow experiments) to Google Drive using DVC

### Key Points

- MLflow always stores experiments locally in `./mlruns` (same as local development)
- After training/evaluation, use `dvc add mlruns` and `dvc push` to sync to Google Drive
- On subsequent runs, `dvc pull` will restore your complete experiment history
- This ensures MLflow data from Colab is accessible on your local machine and vice versa

## Data Version Control (DVC)

This template uses DVC to version control large files and sync them with Google Drive. DVC tracks:
- **Training/test data** (`data/` directory)
- **Model weights** (`models/` directory)  
- **MLflow artifacts** (`mlruns/` directory) - experiment tracking data

This approach keeps your Git repository lightweight while ensuring all important assets are backed up to the cloud.

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

#### 4. Track Data, Models, and MLflow Artifacts with DVC

```bash
# Add data files to DVC tracking
dvc add data/X_train.npy data/y_train.npy data/X_test.npy data/y_test.npy

# Add model weights to DVC tracking
dvc add models/default.keras

# Add MLflow artifacts (experiment tracking) to DVC tracking
dvc add mlruns

# Commit DVC files to Git
git add data/*.dvc data/.gitignore models/*.dvc models/.gitignore mlruns.dvc
git commit -m "Track data, models, and MLflow artifacts with DVC"

# Push everything to Google Drive
dvc push
```

#### 5. Pulling Data on Another Machine or in Colab

When you clone the repository or work in Google Colab, pull all tracked files:

```bash
dvc pull
```
