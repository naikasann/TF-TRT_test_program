import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.applications.resnet50 import ResNet50
import numpy as np
import time
import os

from tftrtmodel.tftrtmodel import TFTRTmodel

keras_model = ResNet50(weights="imagenet")
keras_model.save("resnet50")
tftrt_model = TFTRTmodel(input_model="resnet50",
                            tensorrt_model="resnet50_tensorRT",
                            precision_mode="fp32",
                            batch_size=1,
                            max_workspace_size_bytes=4000000000)
# dammy input
for i in range(100):
    # for multi input layer
    #inputs = {"input_1":x}
    x = np.random.uniform(size=(24, 224, 224, 3)).astype(np.float32)
    y = tftrt_model.predict(x,output_layer="predictions")
    print("{} : output : {}".format(i, np.argmax(y)))
