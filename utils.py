"""
Various utilities
"""

import numpy as np


def print_keras_model(keras_model):
    for layer in keras_model.layers:
        print("layer.get_config():", layer.get_config()['name'])
        weights = layer.get_weights()
        if len(weights) == 2:
            print("layer.get_weights():", weights[0].shape, weights[1].shape)


def compare_weight(keras_model, pytorch_model, weight_name='conv2d_1.weight'):
    for name, param in pytorch_model.named_parameters():
        print(name)
        if name == weight_name:
            pyt_weight = param.detach().numpy()
            print("pyt_weight.shape:", pyt_weight.shape)
    for layer in keras_model.layers:
        if layer.get_config()['name'] == weight_name:
            if weight_name.split()[1] == 'weight':
                keras_weight = layer.get_weights()[0]
                keras_weight = np.transpose(keras_weight, (3, 2, 0, 1))
            elif weight_name.split()[1] == 'bias':
                keras_weight = layer.get_weights()[1]
    print("weight_dis", pyt_weight - keras_weight)

