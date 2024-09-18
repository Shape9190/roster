import pytest
# TODO: add necessary import
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from ml.data import process_data




#collect data
data_path = os.path.join("./data", "census.csv")
data = pd.read_csv(data_path,delimiter=",")

#Variables
train, test = train_test_split(data, test_size=0.20)
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]





# TODO: implement the first test. Change the function name and input as needed
def test_shape():
    """
    # Ensure data is without null values
    """
    assert data.shape == data.dropna().shape
   


# TODO: implement the second test. Change the function name and input as needed
def test_process_data():
    """
    # Confirm process_data function performing as intended
    """

    X_train, y_train, encoder, lb = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=True
    )

   

    assert X_train.shape[0] > 0
    assert y_train.shape[0] > 0

# TODO: implement the third test. Change the function name and input as needed
def test_percentages():
    """
    # check intended percentage of rows/data is delegated
    """
    train_percent = int(data.shape[0] * 0.80)
    test_percent =  int(data.shape[0] * 0.20)

    assert abs(train.shape[0] - train_percent) <= 1
    assert abs(test.shape[0] - test_percent) <= 1