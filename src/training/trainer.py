"""
Training utilities
"""
import time
from src.models.flat_classifier import FlatClassifier


def train_model(df_train, hyperparameters=None):
    """Train a model on training data"""
    if hyperparameters is None:
        hyperparameters = {
            "max_iter": 1000,
            "random_state": 42,
            "solver": "lbfgs",
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
        }
    
    training_start = time.time()
    classifier = FlatClassifier(embedding_model_name=hyperparameters.get("embedding_model", 
                                                                        "paraphrase-multilingual-MiniLM-L12-v2"))
    classifier.train(df_train)
    training_time = time.time() - training_start
    
    return classifier, training_time

