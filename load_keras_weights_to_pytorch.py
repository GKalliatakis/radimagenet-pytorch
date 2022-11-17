"""
Main function for loading Keras-based RadImageNet weights into PyTorch models

References:
    - https://github.com/AgCl-LHY/Weights_Keras_2_Pytorch/blob/main/weights_keras2pytorch.py
    - https://gist.github.com/rAm1n/d848269f69dee431c75331035d282710
    - https://gereshes.com/2019/06/24/how-to-transfer-a-simple-keras-model-to-pytorch-the-hard-way/
"""

import os
import torch
import numpy as np

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet50
from keras.applications.densenet import DenseNet121
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2

from torch import nn
from torchvision.models import resnet50, densenet121, inception_v3
import timm


def fetch_base_torchvision_model(model_name, image_size,
                                 main_keras_weights_dir='./keras_weights',
                                 main_pytorch_weights_dir='./pytorch_weights'):
    """
    Fetches the base PyTorch model from the torchvision.models subpackage
    which contains definitions of models for addressing different tasks.

    model_name (str): name of the base torchvision model to be fetched
    image_size (int): size of the input image (relevant only for Keras models)
    main_keras_weights_dir (str): main directory that holds the downloaded Keras RadImageNet weights
    main_pytorch_weights_dir (str): main directory that will store the (converted) PyTorch weights
    :returns
        pytorch_base_model, pytorch_checkpoint_path, keras_base_model, keras_checkpoint_path
    """
    if not (model_name in {'resnet50', 'densenet121', 'inceptionv3', 'inception_resnet_v2'}):
        raise ValueError("Wrong model name `{}`. "
                         "Available options are: "
                         "['resnet50', 'densenet121', 'inceptionv3', 'inception_resnet_v2']".format(model_name))

    if model_name == 'resnet50':
        pytorch_base_model = resnet50(pretrained=False)
        # pytorch_base_model = nn.Sequential(*list(pytorch_base_model.children())[:-2])  # this will change names
        pytorch_base_model.fc = nn.Identity()  # removing the dense fully connected layer
        keras_checkpoint_path = os.path.join(main_keras_weights_dir, 'RadImageNet-ResNet50_notop.h5')
        pytorch_checkpoint_path = os.path.join(main_pytorch_weights_dir, 'RadImageNet-ResNet50_notop_torch.pth')
        keras_base_model = ResNet50(input_shape=(image_size, image_size, 3),
                                    include_top=False, pooling='avg')

    elif model_name == 'densenet121':
        pytorch_base_model = densenet121(pretrained=False)
        pytorch_base_model.classifier = nn.Identity()  # removing the dense fully connected layer
        keras_checkpoint_path = os.path.join(main_keras_weights_dir, 'RadImageNet-DenseNet121_notop.h5')
        pytorch_checkpoint_path = os.path.join(main_pytorch_weights_dir, 'RadImageNet-DenseNet121_notop_torch.pth')
        keras_base_model = DenseNet121(input_shape=(image_size, image_size, 3),
                                       include_top=False, pooling='avg')

    elif model_name == 'inceptionv3':
        pytorch_base_model = inception_v3(pretrained=False)
        pytorch_base_model.fc = nn.Identity()  # removing the dense fully connected layer
        keras_checkpoint_path = os.path.join(main_keras_weights_dir, 'RadImageNet-InceptionV3_notop.h5')
        pytorch_checkpoint_path = os.path.join(main_pytorch_weights_dir, 'RadImageNet-InceptionV3_notop_torch.pth')
        keras_base_model = InceptionV3(input_shape=(image_size, image_size, 3),
                                       include_top=False, pooling='avg')

    elif model_name == 'inception_resnet_v2':
        pytorch_base_model = timm.create_model(model_name='inception_resnet_v2', pretrained=False)
        keras_checkpoint_path = os.path.join(main_keras_weights_dir, 'RadImageNet-IRV2_notop.h5')
        pytorch_checkpoint_path = os.path.join(main_pytorch_weights_dir, 'RadImageNet-IRV2_notop_torch.pth')
        keras_base_model = InceptionResNetV2(input_shape=(image_size, image_size, 3),
                                             include_top=False, pooling='avg')

    keras_base_model.load_weights(keras_checkpoint_path)

    number_of_keras_parameters = keras_base_model.count_params()
    number_of_pytorch_parameters = sum(p.numel() for p in pytorch_base_model.parameters() if p.requires_grad)
    print('[INFO] Number of total params: '
          'Keras model-> {:,}, PyTorch model-> {:,}'.format(number_of_keras_parameters, number_of_pytorch_parameters))

    return pytorch_base_model, pytorch_checkpoint_path, keras_base_model, keras_checkpoint_path


def keras_to_pytorch(keras_model, pytorch_model, pytorch_checkpoint_path):
    """
    Extracts the original Keras weights (from a given Keras model) into a dictionary which is then used
    to convert the original weight values into torch tensors and
    that will then replace original data of the PyTorch model.
    Finally, it saves the state dictionary of the PyTorch model to a disk file.

    keras_model: a Keras model loaded with trained RadImageNet weights
    pytorch_model: a PyTorch model (with no pretrained weights)
    pytorch_checkpoint_path (str): the full path of the PyTorch checkpoint
    """
    keras_weight_dict = dict()
    for layer in keras_model.layers:
        if type(layer) is keras.layers.convolutional.Conv2D:
            if len(layer.get_weights()) >= 1:
                keras_weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0],
                                                                                         (3, 2, 0, 1))
            if len(layer.get_weights()) >= 2:
                keras_weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.Dense:
            if len(layer.get_weights()) >= 1:
                keras_weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0],
                                                                                         (1, 0))
            if len(layer.get_weights()) >= 2:
                keras_weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.DepthwiseConv2D:
            if len(layer.get_weights()) >= 1:
                keras_weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0],
                                                                                         (2, 3, 0, 1))
            if len(layer.get_weights()) >= 2:
                keras_weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
        elif type(layer) is keras.layers.BatchNormalization:
            if len(layer.get_weights()) >= 1:
                keras_weight_dict[layer.get_config()['name'] + '.weight'] = layer.get_weights()[0]
            if len(layer.get_weights()) >= 2:
                keras_weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]
            if len(layer.get_weights()) >= 3:
                keras_weight_dict[layer.get_config()['name'] + '.running_mean'] = layer.get_weights()[2]
            if len(layer.get_weights()) >= 4:
                keras_weight_dict[layer.get_config()['name'] + '.running_var'] = layer.get_weights()[3]
        elif type(layer) is keras.layers.ReLU:
            pass
        elif type(layer) is keras.layers.Dropout:
            pass
    pytorch_state_dict = pytorch_model.state_dict()
    # print('PyTorch dict. len: {}, Keras dict. len: {}'.format(len(pytorch_state_dict), len(keras_weight_dict)))
    # print(keras_weight_dict.keys())
    if len(pytorch_state_dict) != len(keras_weight_dict):
        print("[WARNING] The two weight dictionaries are not equal! "
              "PyTorch: {} | Keras: {}".format(len(pytorch_state_dict), len(keras_weight_dict)))
    values = list(keras_weight_dict.values())
    i = 0
    for name, param in pytorch_model.named_parameters():
        # print('{}: `{}`'.format(i, name))  # uncomment for added verbosity
        param.data = torch.tensor(values[i])
        i += 1

    torch.save(pytorch_model.state_dict(), pytorch_checkpoint_path)
    print("[INFO] PyTorch checkpoint with the converted RadImageNet weights has been successfully "
          "created @ `{}`".format(pytorch_checkpoint_path))

    # for key in pytorch_state_dict.keys():
    #     print(key)
    #     if 'num_batches_tracked' in key:
    #         continue
    #     pytorch_state_dict[key] = torch.from_numpy(keras_weight_dict[key])
    # pytorch_model.load_state_dict(pytorch_state_dict)
    # return pytorch_model


if __name__ == '__main__':
    # options: ['resnet50', 'densenet121', 'inceptionv3', 'inception_resnet_v2']
    model_name = 'resnet50'

    pytorch_base_model, pytorch_checkpoint_path, \
    keras_base_model, keras_checkpoint_path = fetch_base_torchvision_model(model_name=model_name,
                                                                           image_size=224)

    keras_to_pytorch(keras_base_model, pytorch_base_model, pytorch_checkpoint_path)
