import os
import argparse

import numpy as np
import tensorflow as tf
import mlflow
import mlflow.tensorflow

from utils import load_config, generate_sample_data
from model.sample import SampleNeuralNetwork


def train_model(config):
    '''Train the sample neural network model using configuration'''

    # Extract parameters from config
    model_config = config['model']
    training_config = config['training']
    data_config = config['data']
    paths_config = config['paths']
    mlflow_config = config['mlflow']

    model_name = model_config['name']
    input_dim = model_config['input_dim']
    output_dim = model_config['output_dim']
    activation = model_config['activation']

    learning_rate = training_config['learning_rate']
    epochs = training_config['epochs']
    batch_size = training_config['batch_size']
    n_samples = training_config['n_samples']
    validation_split = training_config['validation_split']

    data_dir = data_config['data_dir']
    train_seed = data_config['train_seed']

    model_dir = paths_config['model_dir']
    experiment_name = mlflow_config['experiment_name']
    tracking_uri = mlflow_config['tracking_uri']

    mlflow.set_experiment(experiment_name)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    print('Generating training data...')
    X_train, y_train = generate_sample_data(
        n_samples=n_samples,
        input_dim=input_dim,
        output_dim=output_dim,
        seed=train_seed
    )

    print('Saving sample data...')
    os.makedirs(data_dir, exist_ok=True)
    np.save(os.path.join(data_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(data_dir, 'y_train.npy'), y_train)

    with mlflow.start_run():
        mlflow.log_params({
            'input_dim': input_dim,
            'output_dim': output_dim,
            'activation': activation,
            'learning_rate': learning_rate,
            'epochs': epochs,
            'batch_size': batch_size,
            'n_samples': n_samples,
            'validation_split': validation_split,
            'train_seed': train_seed
        })

        print('Creating model...')
        model = SampleNeuralNetwork(
            input_dim=input_dim,
            output_dim=output_dim,
            activation=activation,
            name=model_name
        )

        loss = 'binary_crossentropy' if output_dim == 1 else 'categorical_crossentropy'
        metrics = ['accuracy']

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss=loss,
            metrics=metrics
        )

        print(f'Model architecture:')
        model.summary()

        print(f'Training model...')
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )

        final_loss = history.history['loss'][-1]
        final_acc = history.history['accuracy'][-1]
        final_val_loss = history.history['val_loss'][-1]
        final_val_acc = history.history['val_accuracy'][-1]

        mlflow.log_metrics({
            'final_train_loss': final_loss,
            'final_train_accuracy': final_acc,
            'final_val_loss': final_val_loss,
            'final_val_accuracy': final_val_acc
        })

        for epoch in range(epochs):
            mlflow.log_metrics({
                'train_loss': history.history['loss'][epoch],
                'train_accuracy': history.history['accuracy'][epoch],
                'val_loss': history.history['val_loss'][epoch],
                'val_accuracy': history.history['val_accuracy'][epoch]
            }, step=epoch)

        print(f'Saving model...')
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, model_name)
        model.save(model_path)
        print(f'Model saved to: {model_path}')

        print(f'Logging model to MLflow...')
        mlflow.tensorflow.log_model(model, artifact_path='model')

        run_id = mlflow.active_run().info.run_id
        print(f'Model logged to MLflow with run_id: {run_id}')

        print(f'Training completed!')
        print(f'Final training accuracy: {final_acc:.4f}')
        print(f'Final validation accuracy: {final_val_acc:.4f}')

        return run_id, model_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train Sample Neural Network'
    )
    parser.add_argument(
        '--config', type=str, default='../configs/default.yaml', help='Path to configuration file'
    )

    args = parser.parse_args()

    print(f'Loading configuration from: {args.config}')
    config = load_config(args.config)

    print('Starting training process...')
    train_model(config)
