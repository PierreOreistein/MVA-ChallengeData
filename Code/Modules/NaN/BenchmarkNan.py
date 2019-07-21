def BenchmarkNan(df):
    """Replace NaN by Benchmark."""

    # Compute the inverse of the distance
    distance_inv_df = (1. / df.filter(regex='^distance*', axis=1)).values

    # Extract the value at the nearest station
    values_df = df.filter(regex='value_*', axis=1)

    # Compute the benchmark
    numer_df = (distance_inv_df * values_df).sum(axis=1)
    denom_df = (distance_inv_df * (values_df != 0)).sum(axis=1)

    # Compute the benchmark
    benchmark_df = numer_df / denom_df

    # Replace Nan value by benchmark
    df.fillna(benchmark_df, inplace=True)

    return df
