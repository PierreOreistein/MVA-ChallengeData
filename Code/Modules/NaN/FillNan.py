def FillNan(df, method='linear'):
    """Fill Nan values of X_train of X_test with an interpolation."""

    # Interpolate the missing value thanks to interpolation for each station_id
    columns_to_interpolate = ["value_" + str(i) for i in range(10)]

    # Group samples by station_id
    g_stations_df = df.groupby(["station_id"])

    # Loop over the different groups
    for station_id, g in g_stations_df:

        # Interpolate the value of columns_to_interpolate
        values = g[columns_to_interpolate]
        values.interpolate(inplace=True, method=method,
                           limit_direction='forward', axis=0)
        values.interpolate(inplace=True, method=method,
                           limit_direction='backward', axis=0)

        # Replace the values of df
        df.loc[g.index, columns_to_interpolate] = values.values

    return df
