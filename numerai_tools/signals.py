from math import exp, isfinite
from statistics import fmean
from typing import Tuple, Optional

from numerai_tools.scoring import (
    filter_sort_index,
    filter_sort_top_bottom,
    gaussian,
    neutralize,
    spearman_correlation,
    tie_kept_rank,
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


def neutral_churn(
    s1: pd.Series,
    s2: pd.Series,
    neutralizers1: pd.DataFrame,
    neutralizers2: pd.DataFrame,
) -> float:
    """Calculate churn after neutralizing each era's predictions.

    Each prediction is tie-kept ranked, Gaussianized, and neutralized against
    the corresponding era's neutralizers. Churn is then calculated between the
    two neutral residuals as 1 minus their Spearman correlation.

    Arguments:
        s1: pd.Series - predictions from the first era
        s2: pd.Series - predictions from the second era
        neutralizers1: pd.DataFrame - first-era neutralizers
        neutralizers2: pd.DataFrame - second-era neutralizers

    Returns:
        float - the churn between the neutralized predictions
    """
    return churn(
        _neutralize_signal(s1, neutralizers1),
        _neutralize_signal(s2, neutralizers2),
    )


def _neutralize_signal(
    signal: pd.Series,
    neutralizers: pd.DataFrame,
) -> pd.Series:
    signal, neutralizers = filter_sort_index(signal, neutralizers)

    return neutralize(
        gaussian(tie_kept_rank(signal.to_frame())),
        neutralizers,
    ).iloc[:, 0]


def neutral_churn_penalty(
    neutral_churn: float,
    threshold: float = 0.1,
    scaling_factor: float = 20.0,
) -> float:
    """Calculate the fraction of a positive payout retained after a neutral
    churn penalty.

    The retained fraction is ``min(1, 2 / (1 + exp(scaling_factor *
    (neutral_churn - threshold))))``. Callers should apply the returned fraction
    only to positive payouts; burns are not penalized further.

    Arguments:
        neutral_churn: float - post-neutralization churn in the range [0, 2]
        threshold: float - churn through which the full payout is retained
        scaling_factor: float - rate at which the retained payout diminishes

    Returns:
        float - the fraction of a positive payout retained
    """
    assert (
        isfinite(neutral_churn) and 0 <= neutral_churn <= 2
    ), "neutral_churn must be finite and between 0 and 2"
    assert (
        isfinite(threshold) and 0 <= threshold <= 2
    ), "threshold must be finite and between 0 and 2"
    assert (
        isfinite(scaling_factor) and scaling_factor > 0
    ), "scaling_factor must be finite and positive"

    if neutral_churn <= threshold:
        return 1.0

    # This is algebraically equivalent to the capped logistic curve while
    # avoiding overflow for large positive scaling factors.
    decay = exp(-scaling_factor * (neutral_churn - threshold))
    return 2 * decay / (1 + decay)


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


def _clean_signal_submission(
    submission: pd.Series,
    sample_weight: pd.Series,
    dst_id_col: Optional[str] = None,
    dst_signal_col: Optional[str] = None,
) -> Tuple[str, str, pd.Series]:
    ticker_col, signal_col, _, submission_df, _ = validate_submission_signals(
        universe=sample_weight.index.to_frame(),
        submission=submission.reset_index(),
    )
    cleaned_submission = clean_submission(
        universe=sample_weight.index.to_frame(),
        submission=submission_df,
        src_id_col=ticker_col,
        src_signal_col=signal_col,
        dst_id_col=dst_id_col,
        dst_signal_col=dst_signal_col,
        rank_and_fill=True,
    )
    return ticker_col, signal_col, cleaned_submission


def calculate_mean_neutral_churn(
    curr_sub: pd.Series,
    curr_neutralizer: pd.DataFrame,
    curr_sample_weight: pd.Series,
    prev_subs: dict[str, pd.Series],
    prev_neutralizers: dict[str, pd.DataFrame],
    prev_sample_weights: dict[str, pd.Series],
) -> float:
    """Calculate mean neutral churn against recent submissions.

    This uses the same historical lookup and full-universe submission cleaning
    as ``calculate_max_churn_and_turnover``. For a live submission, provide the
    most recent five submissions and their matching era data.

    Arguments:
        curr_sub: pd.Series - current-era submission indexed on tickers/ids
        curr_neutralizer: pd.DataFrame - current-era neutralizers
        curr_sample_weight: pd.Series - current-era sample weights
        prev_subs: dict[str, pd.Series] - recent submissions by datestamp
        prev_neutralizers: dict[str, pd.DataFrame] - neutralizers by datestamp
        prev_sample_weights: dict[str, pd.Series] - sample weights by datestamp

    Returns:
        float - mean neutral churn, or 1.0 when no comparison can be calculated
    """
    curr_ticker_col, curr_signal_col, curr_sub = _clean_signal_submission(
        curr_sub,
        curr_sample_weight,
    )
    neutral_curr_sub = _neutralize_signal(curr_sub, curr_neutralizer)

    neutral_churn_stats = []
    for datestamp in prev_subs:
        _, _, prev_sub = _clean_signal_submission(
            prev_subs[datestamp],
            prev_sample_weights[datestamp],
            dst_id_col=curr_ticker_col,
            dst_signal_col=curr_signal_col,
        )
        neutral_prev_sub = _neutralize_signal(
            prev_sub,
            prev_neutralizers[datestamp],
        )
        try:
            neutral_churn_stats.append(churn(neutral_curr_sub, neutral_prev_sub))
        except AssertionError as error:
            if "does not have enough overlapping ids" in str(error):
                continue
            raise

    return fmean(neutral_churn_stats) if neutral_churn_stats else 1.0


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
    curr_ticker_col, curr_signal_col, curr_sub = _clean_signal_submission(
        curr_sub,
        curr_sample_weight,
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
        _, _, prev_sub = _clean_signal_submission(
            prev_sub,
            prev_sample_weight,
            dst_id_col=curr_ticker_col,
            dst_signal_col=curr_signal_col,
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
