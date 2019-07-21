class NaNDefault(object):
    """Default class for NaN function."""

    def __init__(self, nan=None, hp=None):
        """Define the nan function with the given hyperparameters hp."""

        self.nan = nan
        self.hp = hp
        try:
            self.name = nan.__name__
        except AttributeError:
            self.name = None

    def call(self, df):
        """Call the function nan with the given hp."""

        return self.nan(df, **self.hp)
