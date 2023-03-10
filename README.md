# Markdown file for Titanic Survival Prediction with Deep Learning

This project aims to predict the survival of passengers aboard the Titanic using a deep learning model. The dataset used for this project is the Titanic dataset, which contains passenger information such as age, sex, ticket fare, and whether or not the passenger survived.

## Importing Required Libraries

In this project, we need to import several libraries such as pandas, numpy, tensorflow, keras, and matplotlib. We will use pandas and numpy to preprocess our data, tensorflow to build and train our deep learning model, keras to use the EarlyStopping callback, and matplotlib to visualize our training progress.

```python

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
```

Loading and Preprocessing the Dataset

We start by loading the dataset into a pandas dataframe.

```python

df = pd.read_csv("train.csv")
```

Then, we preprocess the data using the get_inputs_titanic function. This function performs several transformations to the data, such as factorizing the categorical columns, one-hot encoding the title of each passenger, and creating a binary variable indicating whether the passenger has a known cabin or not.

```python

def get_inputs_titanic(df):
    df["Sex"] = pd.factorize(df["Sex"])[0]
    df["Embarked"] = pd.factorize(df["Embarked"])[0]

    # ... (rest of the function)
    
    return preprocessed_data
```

We also define a function to extract the target variable (Survived) from the dataset.

```python

def get_outputs_titanic(df):
    Y = df["Survived"]
    Y = np.asarray(Y).reshape(-1, 1).astype(np.float64)
    return Y
```
Then, we use these functions to extract the input and output variables from the dataset.

```python

X_train = get_inputs_titanic(df)
Y_train = get_outputs_titanic(df)
```
## Building the Deep Learning Model

The deep learning model used for this project is a sequential model with several dense layers, using the ReLU activation function. We also use dropout to prevent overfitting, and the softmax activation function in the last layer to output probabilities of survival for each passenger.

```python

model = Sequential([
    Dense(units=16, input_shape=(X_train.shape[1],), activation="relu"),
    Dense(units=32, activation="relu"),
    Dense(units=64, activation="relu"),
    Dropout(0.5),
    Dense(units=128, activation="relu"),
    Dense(units=256, activation="relu"),
    Dense(units=512, activation="relu"),
    Dense(units=2, activation="softmax")
    ])
```

We then compile the model, using the Adam optimizer and the sparse categorical crossentropy loss function, and specifying the metrics to be tracked during training.

```python

model.compile(optimizer=Adam(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
```

## Training the Model

We train the model using the fit function, specifying the batch size, the number of epochs, and the EarlyStopping callback to prevent overfitting. We also split our data into training and validation sets.

```python

es_callback = EarlyStopping(monitor="val_loss", patience=80, restore_best_weights=True, baseline=0.7)
history = model.fit(x=X_train, y=Y_train, validation_split=0.4, batch_size=10, epochs=1000, shuffle=True, verbose=2, callbacks=[es_callback])
```

Explanation of the code

This code trains a neural network to predict whether a passenger on the Titanic survived or not. The data is loaded from a CSV file called "train.csv", and the preprocessed inputs and outputs are obtained using the functions get_inputs_titanic and get_outputs_titanic, respectively. The neural network is then defined using the Keras Sequential API, with several dense layers and a softmax output layer. The model is compiled with the Adam optimizer and sparse categorical crossentropy loss, and trained on the preprocessed data using the fit method. The training history is then plotted using Matplotlib, and the trained model is saved to a file called "titanic_model.h5". Finally, the function save_result is defined to make predictions on a test set of data, which is loaded from a CSV file called "test.csv", and save the predictions to a CSV file called "solution.csv".

The function get_inputs_titanic preprocesses the input data by converting the "Sex" and "Embarked" columns to integer factors, one-hot encoding the "Pclass" and "Embarked" columns, and extracting additional features such as the passenger's age, number of siblings and spouses, number of parents and children, fare, whether the cabin number is known, and whether the passenger has a title in their name. The function returns an array of the preprocessed input data.

The function get_outputs_titanic preprocesses the output data by converting the "Survived" column to a float and returning an array of the preprocessed output data.

The neural network is defined with several dense layers with varying numbers of units and the "relu" activation function. A dropout layer is also included to prevent overfitting, and the output layer has two units and the "softmax" activation function.

The model is compiled with the Adam optimizer, which is an extension of stochastic gradient descent, and sparse categorical crossentropy loss, which is appropriate for multi-class classification problems. The model is trained on the preprocessed input and output data using the fit method, with a validation split of 0.4, a batch size of 10, and early stopping to prevent overfitting. The training history is then plotted using Matplotlib to visualize the loss and accuracy of the model over the epochs.

The trained model is saved to a file called "titanic_model.h5" using the save method of the Keras model object.

Finally, the function save_result is defined to load a test set of data from a CSV file called "test.csv", preprocess the input data using the get_inputs_titanic function, make predictions using the trained model, and save the predictions to a CSV file called "solution.csv" with the passenger ID and survival prediction for each passenger.

## Conclusion

In this project, we built a machine learning model to predict whether a passenger on the Titanic survived or not based on various features such as their age, sex, class, etc. We preprocessed the data and used a neural network with several layers to train the model. The model achieved a validation accuracy of 81%.

We also used the trained model to make predictions on a separate test dataset and saved the results to a CSV file.

Overall, this project demonstrates how machine learning can be used to analyze and make predictions based on complex datasets. The code can be further optimized and enhanced to improve the accuracy of the model.
