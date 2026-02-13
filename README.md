# General info
This repository includes universal scripts for time series prediction using classical machine learning and deep learning, applicable to any problems.

## Structure of the project

The project contains the following files:
- ``` main.py```: dashboard to change the input data depending on the task
- ``` pred_ML.py ```: functions for different steps, assembled into a Class

A new file for deep learning-based prediction,```pred_NN.py```, will be created soon.

## Detailed files

### ``` main.py ```
Sets the input parameters:
- *dirname*: directory path of the dataset
= *filename*: name of the dataset file
= *feature_names*: list of column names in the dataset
- *target*: name of the target variable
- *ahead*: prediction horizon (time ahead)
- *offset*: time offset used as input data
- *size_split_valtest*: proportion for splitting data into training and validation/test sets 
- *size_split_test*: proportion for splitting data into validation and test sets

### ``` pred_ML.py ```
The class is called Prediction_ML and includes the following methods:
- ```prepare_data```: prepares the dataset
- ```split```: splits data into train/val/set sets
- ```features```: creates input lag features and target variables
- ```train_model```: defines and trains the ML model
- ```predict```: generates predictions on the test set
- ```metrics```: evaluate the model performance 



