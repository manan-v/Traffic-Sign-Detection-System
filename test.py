# Load the trained model
from keras.models import load_model
import numpy as np
import cv2
import pickle
import tensorflow as tf
tf.config.list_physical_devices('GPU')

model = load_model('my_model.h5')

# Load the test image
test_image = cv2.imread('speedLimit60.jpg')

def grayscale(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    return img
def equalize(img):
    img =cv2.equalizeHist(img)
    return img
def preprocessing(img):
    img = grayscale(img)
    img = equalize(img)
    img = img/255
    return img

# Preprocess the test image
preprocessed_image = preprocess(test_image)

# Predict the traffic sign
prediction = model.predict(preprocessed_image)

# Postprocess the prediction
predicted_class = np.argmax(prediction)

# Display the result
print('Predicted traffic sign class:', predicted_class)
cv2.imshow('Test image', test_image)
cv2.waitKey(0)



/Users/mananvadaliya/Downloads/CV project/.DS_Store