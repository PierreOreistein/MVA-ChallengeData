# Math packages
import numpy as np

# Time package
import time

# Progress bar
from tqdm import tqdm

# Neural network
from Modules.Models.NeuralNetworks import *


def CrossValidation(X_train_df, Y_train_df, model_func, model_hp,
                    cv=5, n_jobs=-1):
    """Apply a cross validation to the model."""

    # Initialisation of the time
    start = time.time()

    # Extract all tables as numpy array
    X_train = np.array(X_train_df.iloc[:, 2:].values)
    y_train = np.array(Y_train_df.loc[:, "value"].values).reshape((-1, 1))

    # Shape of data
    n = np.shape(X_train)[0]
    step = n // cv

    def oneFold(k, cv=cv):
        """Execute one fold of the cv."""

        # Index for the training set and testing set
        if k == cv - 1:
            idx_test = np.arange(k * step, n)
        else:
            idx_test = np.arange(k * step, (k + 1) * step)
        idx_train = np.delete(np.arange(0, n), idx_test)

        # Extract the kth X_train and X_test batch
        X_train_k = X_train[idx_train, :]
        y_train_k = y_train[idx_train, :]
        X_test_k = X_train[idx_test, :]
        y_test_k = y_train[idx_test, :]

        # Creation of the model
        model = model_func(**model_hp)

        # Fitting of the model on this batch
        model.fit(X_train_k, y_train_k)

        # Compute the score for this fold
        score_k = model.score(X_test_k, y_test_k)
        print("Score k: ", score_k)

        return score_k

    # Parallelisation of the cv
    all_scores = [oneFold(k) for k in range(cv)]

    # Display the time required
    print("Time of the cross-validation: {:4f}, Score: {:4f}".format(
          time.time() - start, np.mean(all_scores)))

    return all_scores
