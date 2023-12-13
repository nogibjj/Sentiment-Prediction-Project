import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
# Neural Network Classifier
class SentimentAnalysisNN:
    def __init__(self, input_dim):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu", input_shape=(input_dim,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
        )

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def save_model(self, model_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)


class SentimentAnalysisNN_BN:
    def __init__(self, input_dim):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(512, activation="relu", input_shape=(input_dim,)),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(
                    1, activation="sigmoid"
                ),  # Output layer for binary classification
            ]
        )
        self.model.compile(
            optimizer="adam",
            loss="binary_crossentropy",  # Updated loss function
            metrics=["accuracy"],
        )

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def save_model(self, model_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)


class _SentimentAnalysisNN:
    def __init__(self, input_dim):
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(
                    512,
                    activation="relu",
                    input_shape=(input_dim,),
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Dense(
                    256,
                    activation="relu",
                    kernel_regularizer=tf.keras.regularizers.l2(0.01),
                ),
                tf.keras.layers.Dropout(0.6),
                tf.keras.layers.Dense(3, activation="softmax"),
            ]
        )
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )

    def train(self, X_train, y_train, epochs=10, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test)
        return loss, accuracy

    def save_model(self, model_path):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
