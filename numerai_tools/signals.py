from numerai_tools.scoring import (
    filter_sort_index,
    filter_sort_top_bottom,
    spearman_correlation,
)

from typing import List, Tuple, Union, Optional

import pandas as pd


def churn(
    s1: pd.Series,
    s2: pd.Series,
    top_bottom: Optional[int] = None,
) -> float:
    """Calculate the churn between two series. Churn is the proportion of elements
    that are different between the two series.

    For 2 given series with overlapping indices, churn is 1 - Spearman Correlation.
    If top_bottom is provided, the churn is calculated as the average of the % of
    tickers that stay in the top and bottom predictions.

    Arguments:
        s1: pd.Series - the first series to compare
        s2: pd.Series - the second series to compare
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        float - the churn between the two series
    """
    if top_bottom is not None and top_bottom > 0:
        s1_top, s1_bot = filter_sort_top_bottom(s1, top_bottom, False)
        s2_top, s2_bot = filter_sort_top_bottom(s2, top_bottom, False)
        top_overlap = len(s1_top.index.intersection(s2_top.index)) / top_bottom
        bot_overlap = len(s1_bot.index.intersection(s2_bot.index)) / top_bottom
        avg_overlap = (top_overlap + bot_overlap) / 2
        return 1 - avg_overlap

    s1, s2 = filter_sort_index(s1, s2)
    assert s1.std() > 0, "s1 must have non-zero standard deviation"
    assert s2.std() > 0, "s2 must have non-zero standard deviation"
    return 1 - spearman_correlation(s1, s2)
