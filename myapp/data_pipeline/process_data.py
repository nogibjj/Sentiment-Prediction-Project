from utils.preprocessing import load_comments_and_labels, preprocess_comments
from utils.feature_extraction import FeatureExtractor
import numpy as np


def process_data(file_path, vectorizer_file_path):
    # Load and preprocess data
    raw_comments, labels = load_comments_and_labels(file_path)
    processed_comments = preprocess_comments(raw_comments)

    # Feature extraction
    extractor = FeatureExtractor()
    features = extractor.fit_transform(processed_comments)

    # Save the vectorizer to a file
    extractor.save_vectorizer(vectorizer_file_path)

    return features, labels


def process_in_memory_data(raw_comments, labels, vectorizer_file_path):
    # Preprocess data
    processed_comments = preprocess_comments(raw_comments)

    # Feature extraction
    extractor = FeatureExtractor()
    features = extractor.fit_transform(processed_comments)

    # Save the vectorizer to a file
    extractor.save_vectorizer(vectorizer_file_path)

    return features, labels


def save_features_to_file(features, file_path):
    # Save features to a file
    np.savez_compressed(file_path, features=features)
