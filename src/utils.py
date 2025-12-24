import yaml

import numpy as np
import tensorflow as tf


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def generate_sample_data(n_samples=1000, input_dim=10, output_dim=1, seed=42):
    """Generate synthetic data for training"""
    np.random.seed(seed)
    X = np.random.randn(n_samples, input_dim).astype(np.float32)

    if output_dim == 1:
        # Binary classification
        y = (np.sum(X[:, :3], axis=1) > 0).astype(np.float32).reshape(-1, 1)
    else:
        # Multi-class classification
        y = np.random.randint(0, output_dim, size=(n_samples,))
        y = tf.keras.utils.to_categorical(y, num_classes=output_dim)

    return X, y
