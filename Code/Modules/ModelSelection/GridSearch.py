# Persinal packages
from Modules.Utils.ApplyFunctions import *
from Modules.ModelSelection.CrossValidation import CrossValidation
from Modules.NaN.NaNDefault import *

# Set Packages
import itertools
import pandas as pd

# Math Packages
import numpy as np

# Progress bar
from tqdm import tqdm


def dictToCartesianProduct(dct):
    """Transform a dict of lists into a lists of all possible combination."""

    # Convert dict of hyperparameters as list of lists
    dct_l = [dct[key] for key in dct.keys()]

    # Cartesian product of all the possible combination of model hp
    grid_dct_l = [elt for elt in itertools.product(*dct_l)]

    return grid_dct_l


def BuildHpTuples(X_train_df, hyperparameters_NaN_dict, features_l,
                  hyperparameters_models_dict):
    """Build all the possible combination of hyper parameters given as arg."""

    # Grid of hp for all model and kernel functions
    tuples = []

    # Loop over different embedding and data augmentation
    for NaN_func in tqdm(hyperparameters_NaN_dict):

        # Extract the hp of the kernel as a dict
        dct_NaN = hyperparameters_NaN_dict[NaN_func]
        grid_NaN_hp_l = dictToCartesianProduct(dct_NaN)

        for hp_NaN in grid_NaN_hp_l:
            for features_func_group in features_l:

                # Convert hp of the nan as a dict
                keys_list = list(dct_NaN.keys())
                hp_NaN_dct = {keys_list[i]: elt for i, elt in enumerate(hp_NaN)}

                # Definition of the NaN function
                nan = NaNDefault(NaN_func, hp_NaN_dct)

                # Compute the dataset for these functions
                computed_X_train_df = ApplyFunctions(X_train_df, nan,
                                                     features_func_group)

                # Loop over different models
                for model_func in hyperparameters_models_dict.keys():

                    # Compute all possible combinations
                    dct_model = hyperparameters_models_dict[model_func]
                    grid_model_hp_l = dictToCartesianProduct(dct_model)

                    for hp_model in grid_model_hp_l:

                        # Convert hp of the model as a dict
                        keys_list = list(dct_model.keys())
                        hp_model_dct = {keys_list[i]: elt for i,
                                        elt in enumerate(hp_model)}

                        # Definition of the model with the current hp
                        hp_model_dct["shape"] = computed_X_train_df.shape[1] - 2

                        # Add this combination ot tuples
                        tuples.append((nan, hp_NaN_dct,
                                       features_func_group,
                                       model_func, hp_model_dct,
                                       computed_X_train_df))

    return tuples


def subGridSearch(X_train_df, Y_train_df, tuple_i, res_df, cv=5, n_jobs=-1):
    """Execute the CrossValidation on tuple_i of hyperparameters."""

    # Extract the relevant object from tuple_i
    [nan, hp_NaN_dct,
     features_func_group,
     model_func, hp_model_dct,
     computed_X_train_df] = tuple_i

    # Computation of the score trough a Cross Validation
    scores = CrossValidation(computed_X_train_df, Y_train_df,
                             model_func, hp_model_dct, cv=cv, n_jobs=n_jobs)

    # Save the result in the dataFRame res_df
    results = {
        'scores': scores,
        'score': np.mean(scores),

        'NaN_func': NaN.name,
        'NaN_hp': hp_NaN_dct,
        'features_func': features_func_group,
        'model_type': model.name,
        'model_hp': hp_model_dct,
    }
    res_df = res_df.append(results, ignore_index=True)
    res_df.to_csv('./Resultats/grid_search_res.csv', sep='\t')

    # Concatenate the hyperparameters for retruning them
    best_score = score
    best_parameters_names = {"Data Augmentation": {"Function Name": nan.name,
                                                   "Best Parameters": hp_NaN_dct},
                             "Embedding": {"Features Group": features_func_group},
                             "Model": {"Function Name": model.name,
                                       "Best Parameters": hp_model_dct}}
    best_parameters_values = {"Data Augmentation": {"Function": data_aug},
                              "Embedding": {"Function": embedding},
                              "Kernel": {"Function": kernel},
                              "Model": {"Function": model}}

    # Display score and Parameters
    print("Score: {}".format(score))
    print("Best Parameters\n--------")
    print("NaN: ", nan.name, ", hp: ", hp_NaN_dct)
    print("Embedding: ", features_func_group)
    print("Model: ", model.name, ", hp: ", hp_model_dct)
    print("\n\n")

    return best_score, best_parameters_names, best_parameters_values, res_df


def GridSearch(X_train_df, Y_train_df, hps_NaN_dict, features_l,
               hps_models_dict, cv=5, n_jobs=-1, randomise=True):
    """Launch a grid search over different value of the hps."""

    # Compute all the possible combinations of hps
    tuples_hp = BuildHpTuples(X_train_df, hps_NaN_dict,
                              features_l, hps_models_dict)

    # Creates dataframe in which all results will be stored
    # (allows early stopping of grid search)
    pd_res_df = pd.DataFrame()

    # Executes a Cross Validation for all possible tuples
    scores_param = []

    # Randomisation of the tuples
    if randomise:
        np.random.shuffle(tuples_hp)

    for tuple_i in tqdm(tuples_hp):
        [best_score, best_params_n,
         best_params_v, pd_res_df] = subGridSearch(X_train_df, Y_train_df,
                                                   tuple_i,
                                                   pd_res_df, cv=cv,
                                                   n_jobs=n_jobs)
        results = (best_score, best_params_n, best_params_v)
        scores_param.append(results)

    # Extract best scores and parameters
    maxi = 0
    best_params_names = 0
    best_params_values = 0

    for sublist in scores_param:

        if sublist[0] > maxi:
            maxi = sublist[0]
            best_params_names = sublist[1]
            best_params_values = sublist[2]

    # Return result
    return maxi, best_params_names, best_params_values
