def ApplyFunctions(X_train_df, nan, features_func_group):
    """Apply the DataAugmentation and Embedding function to df_dct."""

    # Copy X_train_df
    new_X_train_df = X_train_df.copy()

    # Apply nan function
    new_X_train_df = nan.call(new_X_train_df)

    # Loop over features functions
    for features_func in list(features_func_group):

        # Apply features_func
        new_X_train_df = features_func(new_X_train_df)

    return new_X_train_df
