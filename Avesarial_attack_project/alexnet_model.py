from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, ReLU, MaxPooling2D, Flatten,
    Dense, Dropout, Softmax
)

def create_alexnet(input_shape, num_classes):
    input_layer = Input(shape=input_shape, name="data")

    # Conv1 + ReLU + Pool1
    x = Conv2D(96, kernel_size=11, strides=4, padding="valid", activation=None,
               kernel_initializer="random_normal", bias_initializer="zeros", name="conv1")(input_layer)
    x = ReLU(name="relu1")(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="pool1")(x)

    # Conv2 + ReLU + Pool2
    x = Conv2D(256, kernel_size=5, strides=1, padding="same", activation=None, groups=2,
               kernel_initializer="random_normal", bias_initializer="zeros", name="conv2")(x)
    x = ReLU(name="relu2")(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="pool2")(x)

    # Conv3 + ReLU
    x = Conv2D(384, kernel_size=3, strides=1, padding="same",
               kernel_initializer="random_normal", bias_initializer="zeros", name="conv3")(x)
    x = ReLU(name="relu3")(x)

    # Conv4 + ReLU
    x = Conv2D(384, kernel_size=3, strides=1, padding="same", groups=2,
               kernel_initializer="random_normal", bias_initializer="zeros", name="conv4")(x)
    x = ReLU(name="relu4")(x)

    # Conv5 + ReLU + Pool5
    x = Conv2D(256, kernel_size=3, strides=1, padding="same", groups=2,
               kernel_initializer="random_normal", bias_initializer="zeros", name="conv5")(x)
    x = ReLU(name="relu5")(x)
    x = MaxPooling2D(pool_size=3, strides=2, name="pool5")(x)

    # Flatten for Fully Connected Layers
    x = Flatten(name="flatten")(x)

    # FC6 + ReLU + Dropout
    x = Dense(1024, activation=None, kernel_initializer="random_normal", bias_initializer="zeros", name="fc6")(x)
    x = ReLU(name="relu6")(x)
    x = Dropout(rate=0.5, name="drop6")(x)

    # FC7 + ReLU + Dropout
    x = Dense(1024, activation=None, kernel_initializer="random_normal", bias_initializer="zeros", name="fc7")(x)
    x = ReLU(name="relu7")(x)
    x = Dropout(rate=0.5, name="drop7")(x)

    # FC8 + Softmax for Final Output
    x = Dense(num_classes, activation=None, kernel_initializer="random_normal", bias_initializer="zeros", name="fc8")(x)
    output = Softmax(name="softmax")(x)

    # Create Model
    model = Model(inputs=input_layer, outputs=output, name="AlexNet")
    return model

