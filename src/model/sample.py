import tensorflow as tf


class SampleNeuralNetwork(tf.keras.Model):
    """
    A sample neural network with one dense layer
    """

    def __init__(self, input_dim=10, output_dim=1, activation='sigmoid', name='sample_nn'):
        super(SampleNeuralNetwork, self).__init__(name=name)

        self.dense_layer = tf.keras.layers.Dense(
            units=output_dim,
            activation=activation,
            name='output_layer'
        )

    def call(self, inputs, training=False):
        return self.dense_layer(inputs)
