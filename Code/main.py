# Dataset packages
import pandas as pd

# Utils packages
from Modules.NaN.FillNan import *
from Modules.NaN.BenchmarkNan import *

# Features Creation packages
from Modules.FeaturesCreation.InverseDistance import *
from Modules.FeaturesCreation.addBenchmark import *
from Modules.FeaturesCreation.addSquares import *

# Model packages
from Modules.Models.NeuralNetworks import *

# Model Selection packages
from Modules.ModelSelection.GridSearch import *

# Main Code
if __name__ == '__main__':

	# Loading of the data
	X_train_df = pd.read_csv("./Data/X_train.csv")
	Y_train_df = pd.read_csv("./Data/Y_train.csv")
	X_test_df = pd.read_csv("./Data/X_test.csv")

	# Hyperparameters for the Nan function
	hp_NaN_dct = {
		FillNan: {"method": ['linear']},
		BenchmarkNan: {}
	}

	# List of group of features func to apply
	features_l = [[], [addBenchmark], [addBenchmark, inverseDistance],
				  [addSquares], [inverseDistance, addSquares],
				  [addBenchmark, inverseDistance, addSquares]]

	# Hyperparameters for the model
	hp_model_dct = {
		NN: {
			"dropout": [[0, 0, 0], [0, 0, 0.1], [0, 0, 0.3],
						[0, 0.1, 0.1], [0, 0.3, 0.3], [0.1, 0.1, 0.1]],
			"batch_normalisation": [False, True],
			"nb_neurons_l": [16, 8],
			"epochs": [5, 10],
			"batch_size": [32, 64]}
	}

	# Launch the GridSearch
	[maxi,
	 best_params_names,
	 best_params_values] = GridSearch(X_train_df, Y_train_df,
									  hp_NaN_dct, features_l,
									  hp_model_dct)

	# Make predictions
	# model.makePredictions(X_train_df, X_train, y_train)
