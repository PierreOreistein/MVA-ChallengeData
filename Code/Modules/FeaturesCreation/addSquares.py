def addSquares(df):
    """Add the square of the values of all columns."""

    # Extract all columns of df
    columns = df.columns[2:]

    # Loop over all column in columns
    for col in columns:

        # Add the square
        df[col + str("^2")] = df[col] ** 2

    return df
