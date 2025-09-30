from typing import Union, TypeVar

import pandas as pd

S1 = TypeVar("S1", bound=Union[pd.DataFrame, pd.Series])
S2 = TypeVar("S2", bound=Union[pd.DataFrame, pd.Series])
