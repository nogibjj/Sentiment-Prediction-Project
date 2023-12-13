import sys
from pathlib import Path

# Add parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import scipy.sparse
import tensorflow as tf
from models.naive_bayes_model import NaiveBayesClassifier
from models.neural_network_model import SentimentAnalysisNN_BN as SentimentAnalysisNN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import LabelEncoder
import numpy as np


# Train Naive Bayes Model
def train_nb_model(features, labels, model_path=None):
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # Train Naive Bayes Classifier
    nb_classifier = NaiveBayesClassifier()
    nb_classifier.train(X_train, y_train)

    # Evaluate the model
    y_pred = nb_classifier.predict(X_test)
    print("Naive Bayes Model Accuracy:", accuracy_score(y_test, y_pred))
    print("Naive Bayes Classification Report:\n", classification_report(y_test, y_pred))

    # Save the model
    if model_path:
        nb_classifier.save_model(model_path)

    return nb_classifier


# Encode labels
def encode_labels(labels):
    encoder = LabelEncoder()
    integer_encoded = encoder.fit_transform(labels)
    onehot_encoded = tf.keras.utils.to_categorical(integer_encoded)
    return onehot_encoded


# Encode labels as integers
def encode_labels_int(labels):
    int_labels = [int(label) for label in labels]
    labels_array = np.array(int_labels)
    return labels_array


# Train Neural Network Model
def train_nn_model(features, labels, model_path=None):
    # Convert features to dense format if they are sparse
    if scipy.sparse.issparse(features):
        features = features.toarray()

    # Encode labels as integers
    labels = encode_labels_int(labels)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

    # Train Neural Network Classifier
    input_dim = features.shape[1]
    nn_classifier = SentimentAnalysisNN(input_dim)
    nn_classifier.train(X_train, y_train)

    # Evaluate the model
    loss, accuracy = nn_classifier.evaluate(X_test, y_test)
    print("Neural Network Model Loss:", loss)
    print("Neural Network Model Accuracy:", accuracy)

    # Get predictions
   
    # After getting predictions from the model
    y_pred_prob = nn_classifier.predict(X_test)  # These are probabilities
    y_pred_label = (y_pred_prob > 0.5).astype(
        int
    )  # Convert probabilities to 0 or 1 based on a 0.5 threshold

    # Now y_test and y_pred_label are both binary and can be directly compared
    print(
        "Neural Network Classification Report:\n",
        classification_report(y_test, y_pred_label, zero_division=0),
    )

    # Save the model
    if model_path:
        nn_classifier.save_model(model_path)

    return nn_classifier
