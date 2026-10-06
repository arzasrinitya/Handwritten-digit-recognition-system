# Handwritten Digit Recognition System

A handwritten digit recognition system built using Python and a Convolutional Neural Network (CNN). The system is trained on the MNIST dataset and can recognize handwritten digits from 0 to 9.

## Features

- Trains a CNN model using the MNIST handwritten digit dataset
- Preprocesses handwritten digit images
- Predicts digits from user-provided images
- Displays the predicted digit and prediction confidence
- Saves the trained model for future predictions
- Generates training and validation accuracy/loss graphs

## Project Structure

```text
handwritten-digit-recognition-system/
│
├── model/
│   └── digit_model.keras
│
├── images/
│   ├── accuracy.png
│   ├── loss.png
│   └── test images
│
├── train.py
├── predict.py
├── requirements.txt
├── README.md
└── .gitignore

Technologies Used
Python
TensorFlow
Keras
NumPy
Pillow
Matplotlib
Convolutional Neural Network (CNN)
MNIST Dataset
How It Works
1. Training

The train.py file:

Loads the MNIST dataset.
Normalizes the image pixel values.
Adds the required channel dimension.
Builds a CNN model.
Trains the model for 5 epochs.
Evaluates the model on the test dataset.
Saves the trained model as digit_model.keras.
Generates accuracy and loss graphs.
2. Prediction

The predict.py file:

Loads the trained CNN model.
Takes the path of a handwritten digit image from the user.
Converts the image to grayscale.
Resizes it to 28 × 28 pixels.
Preprocesses and normalizes the image.
Passes the image to the trained model.
Displays the predicted digit and confidence.
Installation

Clone the repository:

git clone https://github.com/arzasrinitya/Handwritten-digit-recognition-system.git

Move into the project folder:

cd Handwritten-digit-recognition-system

Create and activate a virtual environment:

python -m venv venv

On Windows:

venv\Scripts\activate

Install the required libraries:

pip install -r requirements.txt
Train the Model

Run:

python train.py

This trains the CNN and creates the trained model inside the model folder.

Predict a Digit

Run:

python predict.py

When prompted:

Enter image path:

enter the path of your handwritten digit image.

Example:

images/nine.png

The program will display:

Predicted digit: 9
Confidence: XX.XX %
Model Architecture

The CNN consists of:

Convolutional Layer – 32 filters
Max Pooling Layer
Convolutional Layer – 64 filters
Max Pooling Layer
Flatten Layer
Dense Layer – 64 neurons
Output Layer – 10 neurons with Softmax activation

The output layer provides probabilities for digits from 0 to 9.

Results

The model is trained and evaluated using the MNIST dataset. It can correctly recognize clearly written handwritten digits provided as input images.

The images folder contains sample input images and training graphs.

Limitations

The prediction performance can depend on the way a handwritten digit is drawn. Differences in handwriting style, size, positioning, thickness, or shape may affect the prediction because the model is trained on the MNIST dataset.

Future Improvements
Improve preprocessing and image alignment
Increase robustness for different handwriting styles
Add a graphical interface for drawing digits directly
Display prediction probabilities visually
Improve model accuracy and generalization
Author

Srinitya Arza

B.Tech – Information Technology