import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Import dataset
fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# Print shapes of the datasets
print("Train images shape:", train_images.shape)  # (60000, 28, 28)
print("Train labels shape:", train_labels.shape)  # (60000,)
print("Test images shape:", test_images.shape)    # (10000, 28, 28)
print("Test labels shape:", test_labels.shape)    # (10000,)

# Data preprocessing
train_images = train_images / 255.0  # Normalize training images
test_images = test_images / 255.0    # Normalize test images

# Setting up the layers
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker",
               "Bag", "Ankle boot"]
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # Flatten 28x28 images to 784-dimensional vectors
    keras.layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units and ReLU activation
    keras.layers.Dense(10, activation='softmax') # Output layer with 10 units (one for each class) and softmax activation
])

# Compile and optimize the model
model.compile(
    optimizer='adam',  # Adam optimizer
    loss='sparse_categorical_crossentropy',  # Loss function for integer labels
    metrics=['accuracy']  # Track accuracy during training
)

# Training the model
model.fit(train_images, train_labels, epochs=10)

# Evaluating the model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print("Test accuracy:", test_accuracy)
# making predictions

predictions = model.predict(test_images)
print(test_images.shape)
predictions.shape
print(predictions.shape)

# taking a sample to predict if true

"""print(predictions[0])
plt.figure()
plt.imshow(test_images[0])
plt.colorbar()
plt.show()    
"""
# function to plot an image after predicted

def plot_image(i, prediction_array, true_label, img):
    prediction_array, true_label, img =prediction_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(img, cmap=plt.cm.binary)
    predicted_label = np.argmax(prediction_array)
    if predicted_label == true_label:
        color = 'green'
    else:
        color = 'red'
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(prediction_array),
                                         class_names[true_label]),
                                         color=color)        

# functions to create a bar plot of the prediction

def plot_value_array(i, prediction_array, true_label):
    prediction_array, true_label = prediction_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), prediction_array, color = "#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(prediction_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')
    
i = 20
plt.figure(figsize=(6, 3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i], test_labels)
plt.show()   


# testing multiple data

rows = 10
cols = 10

number_images = rows*cols
plt.figure(figsize=(2*2*cols, 2*rows))

for i in range(number_images):
    plt.subplot(rows, 2*cols, 2*i+1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(rows, 2*cols, 2*i+2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()     