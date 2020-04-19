import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from skimage.transform import resize
from keras.models import load_model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# print the data types
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# get the shapes
print('x_train shape:', x_train.shape)
print('y_train shape:', y_train.shape)
print('x_test shape:', x_test.shape)
print('y_test shape:', y_test.shape)

# Take a look at the first image ( at index = 0) in the training dataset
print(x_train[0])

# Show the picture as an image
img = plt.imshow(x_train[0])
plt.show()

# Print the label of the image
print('The label is:', y_train[0])

# Manipulating Y dataset: Convert the labels into a set of10 numbers to input into the neural network
y_train_one_hot = to_categorical(y_train)
y_test_one_hot = to_categorical(y_test)

# Print the new labels in the training dataset
print(y_train_one_hot)

# Print an example of the new labels
print('The one hot label is:', y_train_one_hot[0])

# Normalize the pixels in the images to be values between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# Build the CNN, Create the architecture
model = Sequential()

# Convolution layer
model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(32, 32, 3)))

# MaxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Convolution layer
model.add(Conv2D(32, (5, 5), activation='relu'))

# MaxPooling layer
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(Flatten())

# Create a neural network
model.add(Dense(1000, activation='relu'))
model.add(Dense(10, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# Train the model
hist = model.fit(x_train, y_train_one_hot, batch_size=256, epochs=10, validation_split=0.3)

# Get the models accuracy
show_ACC = model.evaluate(x_test, y_test_one_hot)[1]
print(show_ACC)

# Visualize the models accuracy
plt.plot(hist.history['acc'])
plt.plot(hist.history['val_acc'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.show()

# Visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

# Load the data
img = mpimg.imread('Img/Cat_2a.jpg')

# Show image
plt.imshow(img)
plt.show()

# Resize the image
img_resized = resize(img, (32, 32, 3))
plt.imshow(img_resized)
plt.show()

# Get the probabilities for each class
probabilities = model.predict(np.array([img_resized, ]))

# Print the probabilities
print(probabilities)

number_to_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
index = np.argsort(probabilities[0, :])
print('Most likely class:', number_to_class[index[9]], '--probability:', probabilities[0, index[9]])

# Save the model
#model.save('my_model.h5')

# Load the model
#model = load_model('my_model.h5')
