## Drop Connect - Tensorflow
An implementation of <a href="http://proceedings.mlr.press/v28/wan13.html">Drop-Connect Layer</a> 
in tensorflow 2.x. 
Implementation of layers of Dense, Conv2D, and Wrapper(for all TensorFlow Layers) has been done.

## Demo
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AryaAftab/dropconnect-tensorflow/blob/master/demo/dropconnect_tensorflow_demo.ipynb)
## Install

```bash
$ pip install dropconnect-tensorflow
```

## Usage

### Fully-Connected Network
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from dropconnect_tensorflow import DropConnectDense

# Create Fully-Connected Network
X = tf.keras.layers.Input(shape=(784,))
x = DropConnectDense(units=128, prob=0.2, activation="relu", use_bias=True)(X)
x = DropConnectDense(units=64, prob=0.5, activation="relu", use_bias=True)(x)
y = Dense(10, activation="softmax")(x)

model = tf.keras.models.Model(X, y)


# Hyperparameters
batch_size=64
epochs=20

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),  # Utilize optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

# Train the network
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_split=0.1,
    epochs=epochs)
```

### Convolution Network
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, Input, ReLU, BatchNormalization, Flatten, MaxPool2D
from dropconnect_tensorflow import DropConnectConv2D, DropConnectDense

# Create Convolution Network
X = tf.keras.layers.Input(shape=(28, 28, 1))
x = DropConnectConv2D(filters=64, kernel_size=3, strides=(1, 1), padding='valid', prob=0.1)(X)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D((2,2))(x)
x = DropConnectConv2D(filters=128, kernel_size=3, strides=(1, 1), padding='valid', prob=0.1)(x)
x = BatchNormalization()(x)
x = ReLU()(x)
x = MaxPool2D((2,2))(x)

x = Flatten()(x)
x = DropConnectDense(units=64, prob=0.3, activation="relu", use_bias=True)(x)
y = Dense(10, activation="softmax")(x)

model = tf.keras.models.Model(X, y)


# Hyperparameters
batch_size=64
epochs=20

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),  # Utilize optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

# Train the network
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_split=0.1,
    epochs=epochs)
```

### Wrapper(GRU, LSTM, Dense, Con2D, Conv1D, ...) Network
```python
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, LSTM
from dropconnect_tensorflow import DropConnect

# Create LSTM Network
X = tf.keras.layers.Input(shape=(28,28))

x = DropConnect(LSTM(128, return_sequences=True), prob=0.5)(X)
x = DropConnect(LSTM(128), prob=0.5)(X)
y = Dense(10, activation="softmax")(x)

model = tf.keras.models.Model(X, y)


# Hyperparameters
batch_size=64
epochs=20

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),  # Utilize optimizer
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'])

# Train the network
history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    validation_split=0.1,
    epochs=epochs)
```


## Citations

```bibtex
@inproceedings{wan2013regularization,
  title={Regularization of neural networks using dropconnect},
  author={Wan, Li and Zeiler, Matthew and Zhang, Sixin and Le Cun, Yann and Fergus, Rob},
  booktitle={International conference on machine learning},
  pages={1058--1066},
  year={2013},
  organization={PMLR}
}
```
