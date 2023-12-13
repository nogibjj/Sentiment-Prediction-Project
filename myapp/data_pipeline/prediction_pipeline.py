import sys
from pathlib import Path

# Add parent directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))
import scipy.sparse
import joblib
import tensorflow as tf
from utils.preprocessing import preprocess_comments
from data_pipeline.process_data import process_in_memory_data
from data_pipeline import syn_data_gen as sdg

# File paths
vectorizer = joblib.load("models/tn_vectorizer.pkl")
nb_model = joblib.load("models/tn_naive_bayes_model.pkl")
nn_model = tf.keras.saving.load_model("models/tn_neural_network_model.keras")
syn_vectorizer = joblib.load("models/syn_tn_vectorizer.pkl")
syn_nb_model = joblib.load("models/syn_tn_naive_bayes_model.pkl")
syn_nn_model = tf.keras.saving.load_model("models/syn_tn_neural_network_model.keras")


def prediction_pipeline(text, model_type="nb", synthetic=False):
    if synthetic:
        # Generate synthetic data
        text, syn_labels = sdg.create_balanced_synthetic_dataset(5)
        # print(f"Synthetic text is: {text[0]}")
        # Preprocess the text
        preprocessed_text = preprocess_comments([text[0]])

        # Extract features
        features = syn_vectorizer.transform(preprocessed_text)
        if model_type == "nb":
            # Make a prediction
            prediction = syn_nb_model.predict(features)

        elif model_type == "nn":
            # Convert features to dense format if they are sparse
            if scipy.sparse.issparse(features):
                features = features.toarray()

            # Make a prediction
            prediction = syn_nn_model.predict(features)
        else:
            raise ValueError("Invalid model type")
    else:
        # Preprocess the text
        preprocessed_text = preprocess_comments([text])
        # Extract features
        features = vectorizer.transform(preprocessed_text)
        if model_type == "nb":
            # Make a prediction
            prediction = nb_model.predict(features)
        elif model_type == "nn":
            # Convert features to dense format if they are sparse
            if scipy.sparse.issparse(features):
                features = features.toarray()
            # Make a prediction
            prediction = nn_model.predict(features)
        else:
            raise ValueError("Invalid model type")
    # print(f"Pridiction is: {prediction}")
    return text, interpret_prediction(model_type, prediction)

# Main Function to interpret the prediction
def interpret_prediction(model_type, prediction):
    if model_type == "nb":
        return interpret_nb_prediction(prediction[0])
    elif model_type == "nn":
        return interpret_nn_prediction(prediction[0][0])

    pass
# Function to interpret naive bayes prediction
def interpret_nb_prediction(prediction):
    if prediction == "0" or prediction == 0:
        return "Negative"
    elif prediction == "1" or prediction == 1:
        return "Positive"
    else:
        return "Unknown"
# Function to interpret neural network prediction
def interpret_nn_prediction(prediction):
    if prediction > 0 and prediction < 0.5:
        return "Negative"
    elif prediction >= 0.5:
        return "Positive"
    else:
        return "Unknown"
