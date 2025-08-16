from typing import Tuple, Optional

from numerai_tools.scoring import (
    filter_sort_index,
    filter_sort_top_bottom,
    spearman_correlation,
    generate_neutralized_weights,
)
from numerai_tools.submissions import (
    validate_submission_signals,
    clean_submission,
    remap_ids,
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
        universe -- the universe DataFrame for the current era
        curr_signal_col -- the column name for signal in the current submission
        curr_ticker_col -- the column name for tickers in the current submission

    Returns:
        prev_week_max_churn -- the maximum churn from previous submissions
        prev_week_max_turnover -- the maximum turnover from previous submissions
    """
    universe = universe.reset_index()
    (
        curr_ticker_col,
        curr_signal_col,
        curr_sub,
        _,
    ) = validate_submission_signals(
        universe=universe,
        submission=curr_sub,
    )
    curr_sub_vector = clean_submission(
        universe=universe,
        submission=curr_sub,
        src_id_col=curr_ticker_col,
        src_signal_col=curr_signal_col,
        rank_and_fill=True,
    )
    churn_stats = []
    turnover_stats = []
    neutralized_weights = generate_neutralized_weights(
        curr_sub_vector.to_frame(), curr_neutralizer, curr_weight
    )
    for datestamp in prev_week_subs:
        prev_sub = prev_week_subs[datestamp]
        prev_neutralizer = prev_neutralizers[datestamp]
        prev_weight = prev_sample_weights[datestamp]
        (
            prev_ticker_col,
            prev_signal_col,
            prev_sub,
            _,
        ) = validate_submission_signals(
            universe=universe,
            submission=prev_sub,
        )
        filtered_prev_sub = clean_submission(
            universe=universe,
            submission=prev_sub,
            src_id_col=prev_ticker_col,
            src_signal_col=prev_signal_col,
            dst_id_col=curr_ticker_col,
            dst_signal_col=curr_signal_col,
            rank_and_fill=True,
        )
        prev_neutralizer = (
            remap_ids(
                prev_neutralizer.reset_index(),
                universe,
                str(prev_neutralizer.index.name),
                curr_ticker_col,
            )
            .set_index(curr_ticker_col)
            .filter(like="neutralizer_")
        )
        prev_weight = remap_ids(
            prev_weight.reset_index(),
            universe,
            str(prev_weight.index.name),
            curr_ticker_col,
        ).set_index(curr_ticker_col)[prev_weight.name]
        prev_neutralized_weights = generate_neutralized_weights(
            filtered_prev_sub.to_frame(), prev_neutralizer, prev_weight
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
