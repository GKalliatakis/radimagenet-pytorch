import os
import tensorflow as tf
import numpy as np
import torch
import cv2
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet50, densenet121, inception_v3
from keras.layers.convolutional import Conv2D
import keras
import collections
import timm
import re


def keras_to_pytorch(keras_model, pytorch_model=None):
    weight_dict = dict()
    for layer in keras_model.layers:
        if (type(layer) is Conv2D) and ('0' not in layer.get_config()['name']):
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (3, 2, 0, 1))
            # weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1] as mean
        elif type(layer) is keras.layers.Dense:
            weight_dict[layer.get_config()['name'] + '.weight'] = np.transpose(layer.get_weights()[0], (1, 0))
            weight_dict[layer.get_config()['name'] + '.bias'] = layer.get_weights()[1]

    if pytorch_model:
        pyt_state_dict = pytorch_model.state_dict()
        for key in pyt_state_dict.keys():
            pyt_state_dict[key] = torch.from_numpy(weight_dict[key])
        pytorch_model.load_state_dict(pyt_state_dict)
        return pytorch_model
    return weight_dict


def fetch_torchvision_model(model_name, keras_weights_path='./keras_weights', pytorch_weights_path='./pytorch_weights'):
    if model_name == 'resnet50':
        model = resnet50(num_classes=1)
        keras_weight = os.path.join(keras_weights_path, 'RadImageNet-ResNet50_notop.h5')
        pytorch_weight = os.path.join(pytorch_weights_path, 'RadImageNet-ResNet50_notop_torch.pth')
    elif model_name == 'densenet121':
        model = densenet121(num_classes=1)
        keras_weight = os.path.join(keras_weights_path, 'RadImageNet-DenseNet121_notop.h5')
        pytorch_weight = os.path.join(pytorch_weights_path, 'RadImageNet-DenseNet121_notop_torch.pth')
    elif model_name == 'inceptionv3':
        model = inception_v3(num_classes=1)
        keras_weight = os.path.join(keras_weights_path, 'RadImageNet-InceptionV3_notop.h5')
        pytorch_weight = os.path.join(pytorch_weights_path, 'RadImageNet-InceptionV3_notop_torch.pth')
    elif model_name == 'inception_resnet_v2':
        model = timm.create_model('inception_resnet_v2')
        keras_weight = os.path.join(keras_weights_path, 'RadImageNet-IRV2_notop.h5')
        pytorch_weight = os.path.join(pytorch_weights_path, 'RadImageNet-IRV2_notop_torch.pth')

    return model, keras_weight, pytorch_weight


if __name__ == '__main__':
    print(torch.cuda.get_device_name())
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    tf.config.list_physical_devices('GPU')

    model_name = 'resnet50'
    pytorch_model, keras_checkpoint_path, pytorch_checkpoint_path = fetch_torchvision_model(model_name)



    tf_keras_model = tf.keras.models.load_model(keras_checkpoint_path, compile=False)
    print(tf_keras_model.summary())
    tf_weights = tf_keras_model.get_weights()
    keras_weights_dict = keras_to_pytorch(tf_keras_model)

    values = list(keras_weights_dict.values())
    # print(len(values))
    i = 0
    for name, param in pytorch_model.named_parameters():
        if 'conv' in name:
            param.data = torch.tensor(values[i])
            # print(i, name)
            i += 1

    torch.save(pytorch_model.state_dict(), pytorch_checkpoint_path)

    pytorch_model.load_state_dict(torch.load(pytorch_checkpoint_path))
    # print(pytorch_model)
