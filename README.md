# Tomato Disease Classification: Project Overview
* Created a tool that classify the disease in tomato plants using CNN.
* Built a FastAPI web server which takes in a request of images and returns the predicted disease with the confidence level.

## Code and Resources Used 
**Python Version:** 3.9  
**Packages:** tensorflow, pandas, numpy, matplotlib  
**Install Python Packages:**  ```pip install -r requirements.txt```  
**FastAPI:** https://youtu.be/t6NI0u_lgNo?list=PLeo1K3hjS3ut2o1ay5Dqh-r1kq6ZU8W0M


## Data Preprocessing
Before building a CNN model, I needed to preprocess the data  Here are the steps I took:

- Loaded data into tf.Dataset
- Split the dataset into training, validation and test datasets with a ration of 8:1:1.
- Optimized the performance of input pipelines by catching and prefetching the datasets
- Resized and rescaled the images
- Data augmentation
	- Random Horizontal and Vertical Flips
	- Random Rotation
	- Random Contrast
	

## Model Building 
To train the image classification model, i utilized a convolutional neural network (CNN) architecture implemented using TensorFlow's Keras API. 
The model architecture consists of several Conv2D and MaxPooling2D layers, followed by Flatten and Dropout layers (0.4) for regularization, and finally Dense layers for classification. 
The hyperparameters used in the model training include:
- Optimizer: `Adam` optimizer with its default parameters
- Loss function: `SparseCategoricalCrossentropy` 
- Metrics: `accuracy`
- Batch size: 32
- Number of epochs: 40
- Early stopping: The `EarlyStopping` callback with a patience of 1

## Training and Validation accuracy`and loss
<img src="https://github.com/Gary0417/tomato_disease_classification/blob/data_preprocessing_and_model_building/images/training_and_validation_loss.png">
<img src="https://github.com/Gary0417/tomato_disease_classification/blob/data_preprocessing_and_model_building/images/training_and_validation_accuracy.png">

## Model performance on test dataset
- Loss: 0.1086
- Accuracy 0.9631

## Productionization 
In this step, I exported the trained CNN model to a file on disk built a FastAPI web server and test it using postman application. 
The API takes in a request of images and returns the predicted disease with the confidence level.
