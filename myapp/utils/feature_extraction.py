from sklearn.feature_extraction.text import TfidfVectorizer
import joblib


class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()

    def fit_transform(self, corpus):
        # Fit the vectorizer to the corpus and transform the corpus into TF-IDF features
        return self.vectorizer.fit_transform(corpus)

    def transform(self, new_data):
        # Transform new data using the already fitted vectorizer
        return self.vectorizer.transform(new_data)

    # Save the vectorizer to a file
    def save_vectorizer(self, file_path):
        joblib.dump(self.vectorizer, file_path)
