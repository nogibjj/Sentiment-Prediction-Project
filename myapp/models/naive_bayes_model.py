import os
import joblib
from sklearn.naive_bayes import MultinomialNB


# Naive Bayes Classifier
class NaiveBayesClassifier:
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

    def save_model(self, model_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)
