import tensorflow as tf
from keras.saving import register_keras_serializable


@register_keras_serializable()
class SampleNeuralNetwork(tf.keras.Model):
    """
    A sample neural network with one dense layer
    """

    def __init__(self, input_dim=10, output_dim=1, activation='sigmoid', name='sample_nn', **kwargs):
        super(SampleNeuralNetwork, self).__init__(name=name, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation

        self.dense_layer = tf.keras.layers.Dense(
            units=output_dim,
            activation=activation,
            name='output_layer'
        )

    def call(self, inputs, training=False):
        return self.dense_layer(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'activation': self.activation,
        })
        return config
