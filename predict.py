import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from PIL import Image
import numpy as np

# Load the trained model
model = tf.keras.models.load_model("model/digit_model.keras")

# Enter the path of your handwritten digit image
image_path = input("Enter image path: ")

# Open the image and convert it to grayscale
image = Image.open(image_path).convert("L")

# Resize the image to 28 x 28 pixels
image = image.resize((28, 28))

# Convert the image into a NumPy array
image = np.array(image)

# Invert the image if it has a white background
image = 255 - image

# Save the processed image for inspection
Image.fromarray(image.astype(np.uint8)).save("processed_input.png")

# Normalize pixel values between 0 and 1
image = image / 255.0

# Reshape the image for the CNN
image = image.reshape(1, 28, 28, 1)

# Predict the digit
prediction = model.predict(image, verbose=0)
print("All probabilities:", prediction[0])


# Find the digit with the highest probability and confidence
predicted_digit = np.argmax(prediction)
confidence = prediction[0][predicted_digit] * 100

# Display the result
print("Predicted digit:", predicted_digit)
print("Confidence:", round(confidence, 2), "%")