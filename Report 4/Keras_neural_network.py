# Provide your solution here

# Imports
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Dense, TimeDistributed, Activation
#from keras import layers 
from keras import backend as K

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (7,7) # Make the figures a bit bigger
import numpy as np

# Training parameters
batch_size = 12
num_classes = 90
num_images = 10
epochs = 12

TRAINING_SIZE = 6000
TESTING_SIZE = 1000

from keras.layers import Layer

class Round(Layer):
    def __init__(self, **kwargs):
        super(Round, self).__init__(**kwargs)

    def get_output(self, train=False):
        X = self.get_input(train)
        return K.round(X)

    def get_config(self):
        config = {"name": self.__class__.__name__}
        base_config = super(Round, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    # Data preparation

# Input image dimensions
img_rows, img_cols = 28, 28

# The data, shuffled and split between train and test sets
(x_train_in, y_train_in), (x_test_in, y_test_in) = mnist.load_data()

for i in range(10):
    plt.subplot(1, 10, i + 1)
    plt.axis('off')
    plt.imshow(x_train_in[i], cmap='gray', interpolation='none')
    plt.title(" {}".format(y_train_in[i]))

x_train_in = x_train_in.astype('float32')
x_test_in = x_test_in.astype('float32')
x_train_in /= 255
x_test_in /= 255
    
print(x_train_in.shape, 'train samples')
print(x_test_in.shape, 'test samples')

randomRepetitions = 5

# build training set
x_train = np.zeros((TRAINING_SIZE*(randomRepetitions+1), num_images,img_cols * img_rows), dtype=np.float)
y_train = np.zeros(TRAINING_SIZE*(randomRepetitions+1))
for j in range(TRAINING_SIZE):    

    #just split training images in pairs of 10 images
    for i in range(num_images):

        index = (num_images*j)+i
        x_temp = x_train_in[index] # select next image
        x_temp = x_temp.reshape(1, img_cols * img_rows) # flatten the image
        x_train[j][i]= x_temp
        y_train[j] += y_train_in[index]
        
    # make random combinations of training images  
    for k in range(randomRepetitions):
        for i in range(num_images):

            index = np.random.randint(0, TRAINING_SIZE-1)
            x_temp = x_train_in[index] # select next image
            x_temp = x_temp.reshape(1, img_cols * img_rows) # flatten the image
            x_train[j+TRAINING_SIZE+k*TRAINING_SIZE][i]= x_temp
            y_train[j+TRAINING_SIZE+k*TRAINING_SIZE] += y_train_in[index]      

# build test set
x_test = np.zeros((TESTING_SIZE, num_images, img_cols * img_rows), dtype=np.float)
y_test = np.zeros(TESTING_SIZE)
for j in range(TESTING_SIZE):    
    
    for i in range(num_images):
        #x_temp = x_train_in[np.random.randint(0, TRAINING_SIZE)] # select a random image
        index = (num_images*j)+i
        x_temp = x_test_in[index] # select next image
        x_temp = x_temp.reshape(1, img_cols * img_rows) # flatten the image
        x_test[j][i] = x_temp
        y_test[j]+= y_test_in[index]
    
print(x_train.shape, 'train samples')
print(x_test.shape, 'test samples')

# Convert class vectors to binary class matrices
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_test = keras.utils.to_categorical(y_test, num_classes)
y_train = keras.utils.to_categorical(y_train, 90)
y_test = keras.utils.to_categorical(y_test, 90)

# 5% scoring model
RNN = layers.LSTM
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LAYERS = 1 # need to do 9 additions
MAXLEN = 10


# Model definition
model = Sequential()

# use convolution to look at each image seperately, 10 digits -> 10 dimensions?
model.add(Conv1D(10,img_cols * img_rows, strides=img_cols * img_rows,padding='same',input_shape=x_train.shape[1:])) 

#model.add(Dense(HIDDEN_SIZE, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))        
model.add(Dense(10, activation='relu'))
#model.add(RNN(HIDDEN_SIZE, input_shape=(10,30)))
          
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
#model.add(Dense(10, activation='softmax')) # 10 digits
model.add(Flatten())
model.add(layers.RepeatVector(10)) # do this for all 10 digits?
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    #model.add(Dense(128, activation='relu'))
    model.add(RNN(128, return_sequences=True))
    
# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(90))) # maximum sum


model.add(Flatten())
#model.add(Dense(90, activation='relu'))
model.add(Dense(1, activation=K.relu))


model.compile(loss='mean_squared_error',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# 5% scoring model
RNN = layers.LSTM
HIDDEN_SIZE = 256
BATCH_SIZE = 128
LAYERS = 1 # need to do 9 additions
MAXLEN = 10


# Model definition
model = Sequential()

# use convolution to look at each image seperately, 10 digits -> 10 dimensions?
#model.add(Conv1D(10,img_cols * img_rows, strides=img_cols * img_rows,padding='same',input_shape=x_train.shape[1:])) 

#model.add(Dense(HIDDEN_SIZE, activation='relu', input_shape=x_train.shape[1:]))
model.add(Dense(784, input_shape=(10,784), activation='relu'))
model.add(Dropout(0.2))    # lowered dropout to reduce overfitting
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  
model.add(Dense(10, activation='sigmoid')) # get a value between 1 - 10 for each of the 10 images
#model.add(RNN(HIDDEN_SIZE, input_shape=(10,30)))
          
# As the decoder RNN's input, repeatedly provide with the last hidden state of
# RNN for each time step. Repeat 'DIGITS + 1' times as that's the maximum
# length of output, e.g., when DIGITS=3, max output is 999+999=1998.
#model.add(Dense(10, activation='softmax')) # 10 digits
model.add(Flatten())
model.add(layers.RepeatVector(2)) # do this for all 10 digits? or maximum length 2 (90)
# The decoder RNN could be multiple layers stacked or a single layer.
for _ in range(LAYERS):
    #model.add(Dense(128, activation='relu'))
    model.add(RNN(128, return_sequences=True))
    
# Apply a dense layer to the every temporal slice of an input. For each of step
# of the output sequence, decide which character should be chosen.
model.add(layers.TimeDistributed(layers.Dense(10))) # need to sum 10 elements


model.add(Flatten())
model.add(Dense(90, activation='softmax'))
#model.add(Dense(1, activation=K.relu))


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Training loop
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test)) 
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

predicted_classes = model.predict_classes(x_test)

# Check which items we got right / wrong

correct_indices = np.nonzero(predicted_classes == y_test)[0]
incorrect_indices = np.nonzero(predicted_classes != y_test)[0]

for i in range(10):   
    x = np.zeros((1,10,784))
    index = np.random.randint(0, TESTING_SIZE-1)
    x[0] = x_test[index]
    preds = model.predict(x, verbose=0)
    print("sum: ")
    print(predicted_classes[index])
    print(np.argmax(y_test[index]))
    
    
    