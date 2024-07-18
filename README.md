Logistic Regression Classifier for Cat Images
This project implements a logistic regression classifier to identify cat images. The classifier is trained and tested on a labeled dataset containing cat and non-cat images.

Dataset
The dataset is divided into:

Training Set: Contains a number of images of cats and non-cats.
Test Set: Contains a number of images of cats and non-cats.
The images are of size height_img x height_img x 3 (RGB images).

Data Preprocessing
Reshape Data: The training and test sets are reshaped from (num_samples, height_img, height_img, 3) to (height_img*height_img*3, num_samples).
Normalization: The pixel values are normalized by dividing by 255.
Model Components
Sigmoid Function: Used as the activation function.
Parameter Initialization: Initialize weights w and bias b.
Forward and Backward Propagation: Compute the cost function and its gradient.
Optimization: Optimize weights and bias using gradient descent.
Prediction: Predict labels for given data.
Model Training: Train the logistic regression model and make predictions.
Results
The model is trained and tested, and the accuracies are printed. The learning curve (cost vs. iterations) is plotted to visualize the training process.

Usage
Load Dataset: Load the dataset using load_dataset().
Preprocess Data: Reshape and normalize the data.
Initialize Parameters: Initialize weights and bias.
Compute Gradients: Perform forward and backward propagation to compute gradients.
Optimize Parameters: Optimize weights and bias using gradient descent.
Predict: Use the trained model to make predictions.
Evaluate and Visualize: Evaluate the model and plot the learning curve.
Example Output
Train Accuracy: The percentage of correct predictions on the training set.
Test Accuracy: The percentage of correct predictions on the test set.
Learning Curve: A plot of the cost function vs. the number of iterations.
Issues
If you encounter a runtime warning related to division by zero or invalid value encountered in multiply, add a small epsilon value to avoid log(0).
This project demonstrates the implementation of a logistic regression classifier for image classification tasks. The goal is to provide a simple yet effective method to classify cat images using basic machine learning techniques.
