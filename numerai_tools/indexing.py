from typing import List, Tuple, cast, Any

import numpy as np
import pandas as pd

# leaving this here for backwards compatibility
from numerai_tools.typing import S1, S2


# sometimes when we match up the target/prediction indices,
# changes in stock universe causes some stocks to enter / leave,
# this ensures we don't filter too much
DEFAULT_MAX_FILTERED_INDEX_RATIO = 0.2


def filter_sort_index(
    s1: S1, s2: S2, max_filtered_ratio: float = DEFAULT_MAX_FILTERED_INDEX_RATIO
) -> Tuple[S1, S2]:
    """Filters the indices of the given series to match each other,
    then sorts the indices, then checks that we didn't filter too many indices
    before returning the filtered and sorted series.

    Arguments:
        s1: Union[pd.DataFrame, pd.Series] - the first dataset to filter and sort
        s2: Union[pd.DataFrame, pd.Series] - the second dataset to filter and sort

    Returns:
        Tuple[
            Union[pd.DataFrame, pd.Series],
            Union[pd.DataFrame, pd.Series],
        ] - the filtered and sorted datasets
    """
    ids = s1.dropna().index.intersection(s2.dropna().index)
    # ensure we didn't filter too many ids
    assert len(ids) / len(s1) >= (1 - max_filtered_ratio), (
        "s1 does not have enough overlapping ids with s2,"
        f" must have >= {round(1-max_filtered_ratio,2)*100}% overlapping ids"
    )
    assert len(ids) / len(s2) >= (1 - max_filtered_ratio), (
        "s2 does not have enough overlapping ids with s1,"
        f" must have >= {round(1-max_filtered_ratio,2)*100}% overlapping ids"
    )
    return cast(S1, s1.loc[ids].sort_index()), cast(S2, s2.loc[ids].sort_index())


def filter_sort_index_many(
    inputs: List[Any],
    max_filtered_ratio: float = DEFAULT_MAX_FILTERED_INDEX_RATIO,
) -> List[Any]:
    """Filters the indices of the given list of series to match each other,
    then sorts the indices, then checks that we didn't filter too many indices
    before returning the filtered and sorted series.

    Arguments:
        inputs: List[Union[pd.DataFrame, pd.Series]] - the list of datasets to filter and sort

    Returns:
        List[Union[pd.DataFrame, pd.Series]] - the filtered and sorted datasets
    """
    assert len(inputs) > 0, "List must contain at least one element"
    ids = inputs[0].dropna().index
    for i in range(1, len(inputs)):
        ids = ids.intersection(inputs[i].dropna().index)
    result = [inputs[i].loc[ids].sort_index() for i in range(len(inputs))]
    # ensure we didn't filter too many ids
    for i in range(len(result)):
        assert len(result[i]) / len(inputs[i]) >= (1 - max_filtered_ratio), (
            f"inputs[{i}] does not have enough overlapping ids with the others,"
            f" must have >= {round(1-max_filtered_ratio,2)*100}% overlapping ids"
        )
    return result


def filter_sort_top_bottom(
    s: pd.Series, top_bottom: int
) -> Tuple[pd.Series, pd.Series]:
    """Filters the series according to the top n and bottom n values
    then sorts the index and returns two filtered and sorted series
    for the top and bottom values respectively.

    Arguments:
        s: pd.Series - the data to filter and sort
        top_bottom: int - the number of top n and bottom n values to keep

    Returns:
        Tuple[pd.Series, pd.Series] - the filtered and sorted top and bottom series respectively
    """
    tb_idx = np.argsort(s, kind="stable")
    bot = s.iloc[tb_idx[:top_bottom]]
    top = s.iloc[tb_idx[-top_bottom:]]
    return top.sort_index(), bot.sort_index()


def filter_sort_top_bottom_concat(s: pd.Series, top_bottom: int) -> pd.Series:
    """Similar to filter_sort_top_bottom, but concatenates the top and bottom series
    into 1 series and then sorts the index.

    Arguments:
        s: pd.Series - the data to filter and sort
        top_bottom: int - the number of top n and bottom n values to keep

    Returns:
        pd.Series - the concatenated and sorted series of top and bottom values
    """
    top, bot = filter_sort_top_bottom(s, top_bottom)
    return pd.concat([top, bot]).sort_index()
