def inverse(x):
    """Compute the inverse of x if x not null."""

    if x != 0:
        return 1 / x
    else:
        return 10e10


def inverseDistance(df):
    """Add columns to df corresponding to the inverse of the distances in df."""

    # Loop over all columns of distance
    for i in range(10):

        # Compute new columns
        df["inv_distance_" + str(i)] = df["distance_" + str(i)].apply(lambda x:
                                                                      inverse(x))

    return df
