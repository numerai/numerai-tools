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

    Returns:
        float - the turnover between the two series
    """
    s1, s2 = filter_sort_index(s1, s2)
    turnover = (s1 - s2).abs().sum() / 2
    return turnover


def calculate_max_churn_and_turnover(
    curr_sub: pd.Series,
    curr_neutralizer: pd.DataFrame,
    curr_sample_weight: pd.Series,
    prev_subs: dict[str, pd.Series],
    prev_neutralizers: dict[str, pd.DataFrame],
    prev_sample_weights: dict[str, pd.Series],
) -> Tuple[float, float]:
    """Calculate the maximum churn and turnover of the current submission with respect to previous submissions.
    This function iterates over previous submissions and calculates churn and turnover for each submission
    against the current submission. It expects the following:

        - all submissions, neutralizers, and sample weights are indexed on the same type of tickers/IDs
          (e.g. all numerai_ticker, or all composite_figi, or all etc.)

        - neutralizers and sample weights cover the full universe of their respective eras. This means you
          should avoid removing rows from neutralizers or sample weights before passing them to this function.

    In a live submission environment your submissions are joined on their respective full universes, ranked,
    and then any NaNs are filled with 0.5 before calculating churn and turnover. So, if you provide filtered
    neutralizers or sample weights, your locally calculated churn and turnover may not match the live value.

    Arguments:
        curr_sub: pd.Series - current-era submission indexed on tickers/ids

        curr_neutralizer: pd.DataFrame
            - current-era neutralizers indexed on the same type of tickers/ids.
              We expect these to cover the full universe for the current era.

        curr_sample_weight: pd.Series
            - current-era sample weights indexed on the same type of tickers/ids.
              We expect these to cover the full universe for the current era.

        prev_subs: dict[str, pd.Series]
            - a dictionary mapping datestamps to submissions, where each submission is a
              Series indexed on the same type of tickers/ids as the current
              submission. To calculate churn and turnover for a live submission,
              use the most recent 5 submissions. For diagnostics, just provide the
              last 1 era.

        prev_neutralizers: dict[str, pd.DataFrame]
            - a dictionary mapping datestamps to neutralizers DataFrames where each neutralizers
              DataFrame is indexed on the same type of tickers/ids as the current submission.
              We expect each of these to cover the full universe of their respective eras.

        prev_sample_weights: dict[str, pd.Series]
            - a dictionary mapping datestamps to sample weights where each sample weights
              Series is indexed on the same type of tickers/ids as the current submission.
              We expect each of these to cover the full universe of their respective eras.

    Returns:
        prev_week_max_churn -- the maximum churn from previous submissions
        prev_week_max_turnover -- the maximum turnover from previous submissions
    """
    (
        curr_ticker_col,
        curr_signal_col,
        _,
        curr_sub_df,
        _,
    ) = validate_submission_signals(
        universe=curr_sample_weight.index.to_frame(),
        submission=curr_sub.reset_index(),
    )
    curr_sub = clean_submission(
        universe=curr_sample_weight.index.to_frame(),
        submission=curr_sub_df,
        src_id_col=curr_ticker_col,
        src_signal_col=curr_signal_col,
        rank_and_fill=True,
    )
    churn_stats = []
    turnover_stats = []
    neutralized_weights = generate_neutralized_weights(
        curr_sub.to_frame(),
        curr_neutralizer,
        curr_sample_weight,
        center_and_normalize=True,
    )[curr_sub.name]
    for datestamp in prev_subs:
        prev_sub = prev_subs[datestamp]
        prev_neutralizer = prev_neutralizers[datestamp]
        prev_sample_weight = prev_sample_weights[datestamp]
        (
            prev_ticker_col,
            prev_signal_col,
            _,
            prev_sub_df,
            _,
        ) = validate_submission_signals(
            universe=prev_sample_weight.index.to_frame(),
            submission=prev_sub.reset_index(),
        )
        prev_sub = clean_submission(
            universe=prev_sample_weight.index.to_frame(),
            submission=prev_sub_df,
            src_id_col=prev_ticker_col,
            src_signal_col=prev_signal_col,
            dst_id_col=curr_ticker_col,
            dst_signal_col=curr_signal_col,
            rank_and_fill=True,
        )
        prev_neutralized_weights = generate_neutralized_weights(
            prev_sub.to_frame(),
            prev_neutralizer,
            prev_sample_weight,
            center_and_normalize=True,
        )[prev_sub.name]
        try:
            churn_val = abs(churn(curr_sub, prev_sub))
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
