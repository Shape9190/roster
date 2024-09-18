import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# Adding neccessary imports
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Define model
    model = RandomForestClassifier(min_samples_split=30)
    model.fit(X_train,y_train)

    return model
       
   

    

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : ???
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)

def save_model(model, path):
    """ Serializes model to a file.

    Inputs
    ------
    model
        Trained machine learning model or OneHotEncoder.
    path : str
        Path to save pickle file.
    """
    # Save to pickle file
    with open(path, 'wb') as fp: pickle.dump(model, fp)

def load_model(path):
    """ Loads pickle file from `path` and returns it."""
    #load pickle file
    with open(path, 'rb') as fp:
        model = pickle.load(fp)
    return model


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):

    #Implement the function
    slice_data = data[data[column_name]==slice_value]
    X_slice, y_slice, _, _ = process_data(
        slice_data, categorical_features, label, training=False, encoder=encoder, lb=lb 
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta

