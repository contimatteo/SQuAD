import numpy as np
import pandas as pd

###


class PandasUtils:

    @staticmethod
    def series_to_numpy(df_column: pd.Series, dtype=None) -> np.ndarray:
        if dtype is None:
            return np.array([np.array(xi) for xi in df_column.to_numpy()])
        else:
            return np.array([np.array(xi) for xi in df_column.to_numpy()], dtype=dtype)
