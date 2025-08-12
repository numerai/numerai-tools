from typing import Tuple, Optional

from numerai_tools.submissions import validate_headers_signals, validate_ids_signals
from numerai_tools.scoring import (
    filter_sort_index,
    filter_sort_top_bottom,
    spearman_correlation,
    tie_kept_rank,
    tie_kept_rank__gaussianize__pow_1_5,
    filter_sort_index_many,
    generate_neutralized_weights,
    weight_normalize,
    center,
)

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
    tickers that stay in the top and bottom predictions. This is only relevant when
    the series are rank signals and not portfolio weights.

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
        s1_top, s1_bot = filter_sort_top_bottom(s1, top_bottom)
        s2_top, s2_bot = filter_sort_top_bottom(s2, top_bottom)
        top_overlap = len(s1_top.index.intersection(s2_top.index)) / top_bottom
        bot_overlap = len(s1_bot.index.intersection(s2_bot.index)) / top_bottom
        avg_overlap = (top_overlap + bot_overlap) / 2
        return 1 - avg_overlap

    s1, s2 = filter_sort_index(s1, s2)
    assert s1.std() > 0, "s1 must have non-zero standard deviation"
    assert s2.std() > 0, "s2 must have non-zero standard deviation"
    return 1 - spearman_correlation(s1, s2)


def turnover(
    s1: pd.Series,
    s2: pd.Series,
):
    """Calculate the turnover between two series. Turnover is the total change in weights between
    the two series divided by 2.

    For 2 given series with overlapping indices, join the series on index, fill nans with zeroes
    and calculate turnover as the absolute total difference between the two series divided by 2.
    This is only relevant when the series are portfolio weights and not rank signals.

    Arguments:
        s1: pd.Series - the first series to compare
        s2: pd.Series - the second series to compare
        top_bottom: Optional[int] - the number of top and bottom predictions to use
                                    when calculating the correlation. Results in
                                    2*top_bottom predictions.

    Returns:
        float - the turnover between the two series
    """
    s1, s2 = filter_sort_index(s1, s2)
    turnover = (s1 - s2).abs().sum() / 2
    return turnover


def neutral_weight(
    submission: pd.Series,
    signal_col: str,
    neutralizer: pd.DataFrame,
    weight: pd.Series,
) -> pd.Series:
    s_prime = tie_kept_rank__gaussianize__pow_1_5(submission.to_frame())
    s_prime, neutralizer, weight = filter_sort_index_many(
        [s_prime, neutralizer, weight]
    )
    neutral_weights = generate_neutralized_weights(
        s_prime[signal_col], neutralizer, weight
    )
    neutral_weights = weight_normalize(center(neutral_weights.to_frame()))[0]
    return neutral_weights.sort_index()


def remap_ticker_col(
    predictions: pd.DataFrame,
    universe: pd.DataFrame,
    ticker_col: str,
) -> pd.DataFrame:
    return (
        predictions.join(universe, how="right")
        .reset_index()
        .set_index(ticker_col)
        .sort_index()
    )


def rank_and_fill_signal(
    universe: pd.DataFrame,
    submission: pd.Series,
    signal_col: str,
) -> pd.Series:
    uni_joined_sub = universe.sort_index().join(
        tie_kept_rank(submission.sort_index().to_frame())
    )[[signal_col]]
    filled_sub = uni_joined_sub.fillna(uni_joined_sub.median()).sort_index()
    return filled_sub[signal_col]


def calculate_max_churn_and_turnover(
    curr_sub: pd.DataFrame,
    curr_neutralizer: pd.DataFrame,
    curr_weight: pd.Series,
    prev_week_subs: dict[str, pd.DataFrame],
    prev_neutralizers: dict[str, pd.DataFrame],
    prev_sample_weights: dict[str, pd.Series],
    universe: pd.DataFrame,
    curr_signal_col: str,
    curr_ticker_col: str,
) -> Tuple[float, float]:
    """Calculate the maximum churn and turnover with respect to previous submissions.

    Arguments:
        curr_sub -- the current submission
        curr_neutralizer -- the neutralizer DataFrame for the current submission
        curr_weight -- the sample weights Series for the current submission
        prev_week_subs -- a dictionary of datestamps to submissions
        prev_neutralizers -- a dictionary of datestamps to neutralizers
        prev_sample_weights -- a dictionary of datestamps to sample weights
        universe -- the internal universe DataFrame
        curr_signal_col -- the column name for signal in the current submission
        curr_ticker_col -- the column name for tickers in the current submission

    Returns:
        prev_week_max_churn -- the maximum churn from previous submissions
        prev_week_max_turnover -- the maximum turnover from previous submissions
    """
    curr_sub_vector: pd.Series = rank_and_fill_signal(
        universe,
        curr_sub.reset_index().set_index(curr_ticker_col).sort_index()[curr_signal_col],
        curr_signal_col,
    )
    churn_stats = []
    turnover_stats = []
    neutralized_weights = neutral_weight(
        curr_sub_vector, curr_signal_col, curr_neutralizer, curr_weight
    )
    for datestamp in prev_week_subs:
        prev_sub = prev_week_subs[datestamp]
        prev_neutralizer = prev_neutralizers[datestamp]
        prev_weight = prev_sample_weights[datestamp]
        prev_ticker_col, prev_signal_col = validate_headers_signals(prev_sub)
        prev_universe = universe.reset_index().set_index(prev_ticker_col)
        filtered_prev_sub_df, _ = validate_ids_signals(
            prev_universe.index.to_series(), prev_sub, prev_ticker_col
        )
        # in case the previous submission has a different ticker column,
        # remap the ticker column of prev data to the current ticker column
        filtered_prev_sub = remap_ticker_col(
            filtered_prev_sub_df.set_index(prev_ticker_col),
            universe=prev_universe,
            ticker_col=curr_ticker_col,
        )[curr_signal_col]
        filtered_prev_sub = rank_and_fill_signal(
            universe=universe,
            submission=filtered_prev_sub,
            signal_col=curr_signal_col,
        )
        prev_neutralizer = remap_ticker_col(
            prev_neutralizer,
            universe=prev_universe,
            ticker_col=curr_ticker_col,
        ).filter(like="neutralizer_")
        prev_weight = remap_ticker_col(
            prev_weight.to_frame(),
            universe=prev_universe,
            ticker_col=curr_ticker_col,
        )[prev_weight.name]
        prev_neutralized_weights = neutral_weight(
            filtered_prev_sub, prev_signal_col, prev_neutralizer, prev_weight
        )
        try:
            churn_val = abs(churn(curr_sub_vector, filtered_prev_sub))
        except AssertionError as e:
            if "does not have enough overlapping ids" in str(e):
                continue
        try:
            turnover_val = abs(turnover(neutralized_weights, prev_neutralized_weights))
        except AssertionError as e:
            if "does not have enough overlapping ids" in str(e):
                continue

        churn_stats.append(churn_val)
        turnover_stats.append(turnover_val)
    if len(churn_stats) == 0:
        prev_week_max_churn = 1.0
    else:
        prev_week_max_churn = max(churn_stats)
    if len(turnover_stats) == 0:
        prev_week_max_turnover = 1.0
    else:
        prev_week_max_turnover = max(turnover_stats)
    return prev_week_max_churn, prev_week_max_turnover
