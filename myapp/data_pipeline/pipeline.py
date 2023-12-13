import collect_comments as cc
import train_models as tm
import process_data as pd
import numpy as np
import os
from dotenv import load_dotenv
import sentiment_labeling as sl

# File paths
ms_comment_file_path = "database/microsoft_comments.csv"
am_comment_file_path = "database/amazon_comments.csv"
ms_com_lab_file_path = "database/microsoft_comments_labeled.csv"
am_com_lab_file_path = "database/amazon_comments_labeled.csv"
ms_vec_file_path = "models/microsoft_vectorizer.pkl"
am_vec_file_path = "models/amazon_vectorizer.pkl"
ms_nb_model_file_path = "models/ms_naive_bayes_model.pkl"
am_nb_model_file_path = "models/am_naive_bayes_model.pkl"
ms_nn_model_file_path = "models/ms_neural_network_model.keras"
am_nn_model_file_path = "models/am_neural_network_model.keras"


def pipeline():
    # Load environment variables
    load_dotenv()
    client_id = os.getenv("CLIENT_ID")
    client_secret = os.getenv("CLIENT_SECRET")
    user_agent = os.getenv("USER_AGENT")

    # Collect comments
    cc.collect_comments(
        client_id, client_secret, user_agent, ms_comment_file_path
    )

    # Add sentiment labels to comments
    sl.create_sentiment_labels(ms_comment_file_path, ms_com_lab_file_path)
    #sl.create_sentiment_labels(am_comment_file_path, am_com_lab_file_path)

    # Process data
    ms_features, ms_labels = pd.process_data(ms_com_lab_file_path, ms_vec_file_path)
    #am_features, am_labels = pd.process_data(am_com_lab_file_path, am_vec_file_path)

    # Train nb model
    ms_nb = tm.train_nb_model(ms_features, ms_labels, ms_nb_model_file_path)
   # am_nb = tm.train_nb_model(am_features, am_labels, am_nb_model_file_path)

    # Train nn model
    ms_nn = tm.train_nn_model(ms_features, ms_labels, ms_nn_model_file_path)
   #am_nn = tm.train_nn_model(am_features, am_labels, am_nn_model_file_path)


if __name__ == "__main__":
    pipeline()
