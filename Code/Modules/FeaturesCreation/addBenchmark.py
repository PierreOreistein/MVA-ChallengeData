def addBenchmark(df):
    """Add benchmark to df."""

    # Compute the inverse of the distance
    distance_inv = (1. / df.filter(regex='^distance*', axis=1)).values

    # Extract the value at the nearest station
    values = df.filter(regex='value_*', axis=1)

    # Compute the benchmark
    numer = (distance_inv * values).sum(axis=1)
    denom = (distance_inv * (values != 0)).sum(axis=1)

    # Compute the benchmark
    benchmark = numer / denom
    df["Benchmark"] = benchmark

    return df
