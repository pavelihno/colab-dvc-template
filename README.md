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

MLflow will automatically create a local tracking directory. To view the MLflow UI:

```bash
mlflow server --backend-store-uri sqlite:///db/mlflow.db --registry-store-uri sqlite:///db/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
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
cd src
python train.py --config ../configs/default.yaml
```

The training script will output a **run_id** at the end, which you'll need for evaluation. Example output:

```
Model logged to MLflow with run_id: abc123def456...
```

**Note:** The trained model is saved both locally (in `models/` directory) and logged to MLflow for experiment tracking and easy retrieval.

### Evaluating a Model

Evaluate a trained model using its MLflow run_id (obtained from training):

```bash
cd src
python eval.py --config ../configs/default.yaml --run_id <RUN_ID_FROM_TRAINING>
```

**Note:** The evaluation script loads the model directly from MLflow using the run_id, ensuring you're evaluating the exact model version that was logged during training.

### Viewing Results in MLflow

1. Start the MLflow UI:

    ```bash
    mlflow server --backend-store-uri sqlite:///db/mlflow.db --registry-store-uri sqlite:///db/mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
    ```

2. Open http://localhost:5000 in your browser

3. Select your experiment

4. View and compare different runs with their parameters and metrics

5. Visualize training curves and download models

## Using in Google Colab

Open `notebooks/colab.ipynb` in Google Colab:

1. Upload the notebook to Google Colab
2. Run the first cell to install requirements
3. Mount your Google Drive if needed for data/model storage
4. Run training and evaluation cells

## Data Version Control (DVC)

This template includes DVC for data versioning with Google Drive integration.

### Setting up DVC with Google Drive

```bash
# Initialize DVC (if not already done)
dvc init

# Configure Google Drive as remote storage
dvc remote add -d gdrive gdrive://FOLDER_ID

# Add data files to DVC tracking
dvc add data/X_train.npy data/y_train.npy

# Commit DVC files to Git
git add data/X_train.npy.dvc data/y_train.npy.dvc data/.gitignore .dvc/config
git commit -m "Add dataset with DVC"

# Push data to Google Drive
dvc push
```

### Getting the Google Drive Folder ID

1. Create a folder in your Google Drive for DVC storage
2. Open the folder in your browser
3. Copy the folder ID from the URL: `https://drive.google.com/drive/folders/FOLDER_ID`
4. Use this ID when configuring the DVC remote

### Using DVC in Google Colab

When working in Google Colab, you can easily pull data from your DVC remote:

```python
# In your Colab notebook
!pip install dvc[gdrive]
!dvc pull
```

DVC will authenticate with your Google Drive account and download the tracked data files automatically.
