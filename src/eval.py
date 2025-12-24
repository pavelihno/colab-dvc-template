import argparse
import os
import numpy as np

import mlflow

from utils import load_config, generate_sample_data

from model.sample import SampleNeuralNetwork


def evaluate_model(config, run_id=None):
    '''Evaluate the trained model on test data using configuration'''

    # Extract parameters from config
    model_config = config['model']
    eval_config = config['evaluation']
    data_config = config['data']
    paths_config = config['paths']
    mlflow_config = config['mlflow']

    input_dim = model_config['input_dim']
    output_dim = model_config['output_dim']
    activation = model_config['activation']

    n_samples = eval_config['n_samples']

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, paths_config['data_dir'])
    test_seed = data_config['test_seed']

    experiment_name = mlflow_config['experiment_name']
    tracking_uri = mlflow_config['tracking_uri']

    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    if not run_id:
        raise ValueError('run_id is required to load the model from MLflow')

    model_uri = f'runs:/{run_id}/model'
    print(f'Loading model from MLflow run: {run_id}')
    model = SampleNeuralNetwork(
        input_dim=input_dim,
        output_dim=output_dim,
        activation=activation,
    )
    model = mlflow.tensorflow.load_model(model_uri)

    print('Generating test data...')
    X_test, y_test = generate_sample_data(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=test_seed
    )

    print('Saving sample test data...')
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), y_test)

    print(f'Evaluating model on {n_samples} test samples...')

    if run_id:
        # Continue existing run
        with mlflow.start_run(run_id=run_id):
            results = model.evaluate(X_test, y_test, verbose=1)
            loss, accuracy = results[0], results[1]

            mlflow.log_metrics({
                'test_loss': loss,
                'test_accuracy': accuracy
            })
    else:
        # Create new evaluation run
        with mlflow.start_run(run_name='evaluation'):
            mlflow.log_params({
                'n_test_samples': n_samples,
                'model_run_id': run_id
            })

            results = model.evaluate(X_test, y_test, verbose=1)
            loss, accuracy = results[0], results[1]

            # Log evaluation metrics
            mlflow.log_metrics({
                'test_loss': loss,
                'test_accuracy': accuracy
            })

    print(f'Evaluation Results:')
    print(f'Test Loss: {loss:.4f}')
    print(f'Test Accuracy: {accuracy:.4f}')

    metrics = {
        'test_loss': loss,
        'test_accuracy': accuracy
    }

    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate Sample Neural Network'
    )
    parser.add_argument(
        '--config', type=str, default='../configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--run_id', type=str, required=True,
        help='MLflow run ID to load the trained model from'
    )

    args = parser.parse_args()

    print(f'Loading configuration from: {args.config}')
    config = load_config(args.config)

    print(f'Starting model evaluation...')
    evaluate_model(config, run_id=args.run_id)
