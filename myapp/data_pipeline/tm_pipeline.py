import train_models as tm
import process_data as pd
import syn_data_gen as sdg


# File paths
tn_file_path = "database/amazon.csv"
tn_vec_file_path = "models/tn_vectorizer.pkl"
tn_nb_model_file_path = "models/tn_naive_bayes_model.pkl"
tn_nn_model_file_path = "models/tn_neural_network_model.keras"
syn_vec_file_path = "models/syn_tn_vectorizer.pkl"
syn_nb_model_file_path = "models/syn_tn_naive_bayes_model.pkl"
syn_nn_model_file_path = "models/syn_tn_neural_network_model.keras"


def tm_pipeline():
    # Process real data
    tn_features, tn_labels = pd.process_data(tn_file_path, tn_vec_file_path)

    # Train nb model on real data
    print("Training Naive Bayes Model on Real Data...")
    tm.train_nb_model(tn_features, tn_labels, tn_nb_model_file_path)

    # Train nn model on real data
    print("Training Neural Network Model on Real Data...")
    tm.train_nn_model(tn_features, tn_labels, tn_nn_model_file_path)

    # Generate synthetic data
    syn_comments, syn_labels = sdg.create_balanced_synthetic_dataset()

    # Process synthetic data
    synthetic_features, synthetic_labels = pd.process_in_memory_data(
        syn_comments, syn_labels, syn_vec_file_path
    )

    # Train nb model on synthetic data
    print("Training Naive Bayes Model on Synthetic Data...")
    tm.train_nb_model(synthetic_features, synthetic_labels, syn_nb_model_file_path)

    # Train nn model on synthetic data
    print("Training Neural Network Model on Synthetic Data...")
    tm.train_nn_model(synthetic_features, synthetic_labels, syn_nn_model_file_path)


if __name__ == "__main__":
    tm_pipeline()
