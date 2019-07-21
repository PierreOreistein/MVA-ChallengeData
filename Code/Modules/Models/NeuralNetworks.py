# Math packages
import numpy as np

# Keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import BatchNormalization

# Sklearn
import sklearn.metrics as metrics


class NN(object):
    def __init__(self, shape=1, dropout=0, batch_normalisation=False,
                 nb_neurons_l=16, epochs=15, batch_size=64):
        """Initialisation of the neural network."""

        # Save parameters of the compilator
        self.epochs = epochs
        self.batch_size = batch_size

        # Compute the input shape
        input_shape = (shape, )

        # Extract numbers of neurons
        if type(nb_neurons_l) == int:
            nb_neurons_l = [nb_neurons_l for i in range(3)]

        # Definition of the model
        self.model = Sequential()

        # Dense Layer
        if batch_normalisation:
            self.model.add(BatchNormalization())

        self.model.add(Dense(nb_neurons_l[0], activation="relu",
                             input_shape=input_shape))

        if dropout[0] > 0:
            self.model.add(Dropout(dropout[0]))

        self.model.add(Dense(nb_neurons_l[1], activation="relu"))

        if dropout[1] > 0:
            self.model.add(Dropout(dropout[1]))

        self.model.add(Dense(nb_neurons_l[2], activation="relu"))

        if dropout[2] > 0:
            self.model.add(Dropout(dropout[2]))

        self.model.add(Dense(1, activation="relu"))

        # Definition of the loss function
        self.model.compile(loss='mean_squared_logarithmic_error',
                           optimizer="adam")

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Fitting of the model."""

        if (X_val is None) or (y_val is None):
            self.model.fit(X_train, y_train,
                           epochs=self.epochs, batch_size=self.batch_size,
                           verbose=1)
        else:
            self.model.fit(X_train, y_train,
                           epochs=self.epochs, batch_size=self.batch_size,
                           verbose=1, validation_data=(X_val, y_val))

    def predict(self, X):
        """Predictions for the dataset given in arguument."""

        # Make predictions
        y_pred = self.model.predict(X)

        return y_pred

    def score(self, X, y):
        """Compute the score between the prediction of X and the true y."""

        score = metrics.mean_squared_log_error(y, self.predict(X))

        return score

    def makePredictions(self, X_test_df, X_train, y_train):
        """Compyte the predictions and save the results."""

        # Make predictions
        y_pred_values = self.predict(X_test_df.iloc[:, 2:].values)

        # Clip the predictions to be positives
        y_pred_values = np.where(y_pred_values < 0, 0, y_pred_values)

        # Disaply the scores
        print("Score on the training set: ",
              metrics.mean_squared_log_error(y_train,
                                             self.model.predict(X_train)))

        # Save predictions
        Y_pred_df = X_test_df.ID.to_frame()
        Y_pred_df["value"] = y_pred_values
        Y_pred_df.to_csv("./Results/Predictions_NN.csv", index=False)
