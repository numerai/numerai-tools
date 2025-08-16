from numerai_tools.scoring import tie_kept_rank

import logging
from typing import Tuple, List, Optional

import pandas as pd
import numpy as np

NUMERAI_ALLOWED_ID_COLS = ["id"]
NUMERAI_ALLOWED_PRED_COLS = ["prediction", "probability"]

SIGNALS_ALLOWED_ID_COLS = [
    "ticker",
    "sedol",
    "bloomberg_ticker",
    "composite_figi",
    "numerai_ticker",
]
SIGNALS_ALLOWED_PRED_COLS = ["prediction", "signal"]
SIGNALS_ALLOWED_DATE_COLS = ["friday_date", "date"]
SIGNALS_MIN_TICKERS = 100

CRYPTO_ALLOWED_ID_COLS = ["symbol"]
CRYPTO_ALLOWED_PRED_COLS = ["prediction", "signal"]
CRYPTO_MIN_TICKERS = 100

logger = logging.getLogger(__name__)


def _validate_headers(
    submission: pd.DataFrame,
    expected_id_cols: List[str],
    expected_pred_cols: List[str],
    other_cols: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """Validate the given submission has the right headers.
    It is recommended to use one of the following functions instead of this one:
        - validate_headers_numerai
        - validate_headers_signals

    Arguments:
        submission -- pandas DataFrame of the submission
        expected_id_cols -- list of expected id columns
        expected_pred_cols -- list of expected prediction columns
        other_cols -- optional list of other columns that can be present in the submission

    Return Tuple[str, str]:
        - string name of the id column
        - string name of the prediction column
    """
    expected_headers = [
        [ticker_col, signal_col]
        for ticker_col in expected_id_cols
        for signal_col in expected_pred_cols
    ]
    if other_cols is not None:
        expected_headers += [
            [ticker_col, signal_col, other_col]
            for ticker_col in expected_id_cols
            for signal_col in expected_pred_cols
            for other_col in other_cols
        ]
    columns = submission.columns
    valid_headers = list(columns) in expected_headers
    assert valid_headers, (
        "invalid_submission_headers: headers must be one of"
        f" {expected_id_cols} and one of {expected_pred_cols}"
    )
    return columns[0], columns[1]


def validate_headers_numerai(submission: pd.DataFrame) -> Tuple[str, str]:
    return _validate_headers(
        submission,
        NUMERAI_ALLOWED_ID_COLS,
        NUMERAI_ALLOWED_PRED_COLS,
    )


def validate_headers_signals(submission: pd.DataFrame) -> Tuple[str, str]:
    return _validate_headers(
        submission,
        SIGNALS_ALLOWED_ID_COLS,
        SIGNALS_ALLOWED_PRED_COLS,
        SIGNALS_ALLOWED_DATE_COLS,
    )


def validate_headers_crypto(submission: pd.DataFrame) -> Tuple[str, str]:
    return _validate_headers(
        submission,
        CRYPTO_ALLOWED_ID_COLS,
        CRYPTO_ALLOWED_PRED_COLS,
    )


def validate_values(submission: pd.DataFrame, prediction_col: str) -> None:
    """
    Validates the given submission's values are between 0 and 1 exclusive and
    that the submission have a non-zero standard deviation.

    Arguments:
        submission -- pandas DataFrame of the submission
        prediction_col -- the string name of the prediction column returned by validate_headers
    """
    assert (
        submission[prediction_col].isna().sum() == 0
    ), "invalid_submission_values: submission must not contain NaNs"
    assert (
        submission[prediction_col].between(0, 1).all()
    ), "invalid_submission_values: values must be between 0 and 1 exclusive"
    assert not np.isclose(
        0, submission[prediction_col].std()
    ), "invalid_submission_values: submission must have non-zero standard deviation"


def _validate_ids(
    live_ids: pd.Series, submission: pd.DataFrame, id_col: str, min_tickers: int
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Validates the given submission has no NaNs in the given id column
    and that the submission has a minimum number of non-duplicate ids
    after filtering to the live_ids.

    It is recommended to use one of the following functions instead of this one:
        - validate_ids_numerai
        - validate_ids_signals

    Arguments:
        live_ids -- pandas Series of the live ids or tickers from live universe
        submission -- pandas DataFrame of the submission
        id_col -- the stringn name of the column containing ids or tickers

    Return Tuple[pd.DataFrame, List[str]]:
        - submission indexed on id_col and filtered against live_ids
        - set of invalid tickers (diff between indexed sub and live_ids-joined sub)
    """
    assert (
        not submission[id_col].isna().any()
    ), f"invalid_submission_ids: Submission must not contain NaNs in the {id_col} column."

    index_sub = submission.copy()
    index_sub[id_col] = index_sub[id_col].astype(str)

    live_ids = live_ids.astype(str)
    live_sub = index_sub[index_sub[id_col].isin(live_ids)].sort_values(id_col)
    assert (
        not live_sub[id_col].duplicated().any()
    ), f"invalid_submission_ids: Duplicates detected in {id_col} for live period."

    # join on live_ids and ensure min tickers reached
    assert len(live_sub) >= min_tickers, (
        "invalid_submission_ids: Not enough stocks submitted."
        " Are you using the latest live ids or live universe?"
    )

    invalid_tickers = list(set(index_sub[id_col]).difference(set(live_sub[id_col])))
    return live_sub, invalid_tickers


def validate_ids_numerai(
    live_ids: pd.Series, submission: pd.DataFrame, id_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    return _validate_ids(live_ids, submission, id_col, len(live_ids))


def validate_ids_signals(
    live_ids: pd.Series, submission: pd.DataFrame, id_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    return _validate_ids(live_ids, submission, id_col, SIGNALS_MIN_TICKERS)


def validate_ids_crypto(
    live_ids: pd.Series, submission: pd.DataFrame, id_col: str
) -> Tuple[pd.DataFrame, List[str]]:
    return _validate_ids(live_ids, submission, id_col, CRYPTO_MIN_TICKERS)


def validate_submission_numerai(
    universe: pd.Series, submission: pd.DataFrame
) -> Tuple[str, str, pd.DataFrame, List[str]]:
    """Validate the headers, ids, and values for a submission.

    Arguments:
        universe: pd.DataFrame - the live universe of ids on which the predictions are based
        submission: pd.DataFrame - the predictions to validate

    Returns:
        Tuple[str, str, pd.DataFrame, List[str]] - the validated ticker column, signal column,
                                                   filtered submission, and list of invalid tickers
    """
    ticker_col, signal_col = validate_headers_numerai(submission)
    filtered_sub, invalid_tickers = validate_ids_numerai(
        universe, submission, ticker_col
    )
    validate_values(filtered_sub, signal_col)
    return ticker_col, signal_col, filtered_sub, invalid_tickers


def validate_submission_signals(
    universe: pd.DataFrame, submission: pd.DataFrame
) -> Tuple[str, str, pd.DataFrame, List[str]]:
    """Validate the headers, ids, and values for a submission.

    Arguments:
        universe: pd.DataFrame - the live universe of ids on which the predictions are based
        submission: pd.DataFrame - the predictions to validate

    Returns:
        Tuple[str, str, pd.DataFrame, List[str]] - the validated ticker column, signal column,
                                                   filtered submission, and list of invalid tickers
    """
    # drop data_type and date columns if they exist
    if "data_type" in submission.columns:
        logger.warning(
            "data_type column found in Signals submission. This is deprecated and support will be removed in the future. "
            "Please remove the data_type column from your Signals submission."
        )
        submission = submission.drop(columns=["data_type"], errors="ignore")
    ticker_col, signal_col = validate_headers_signals(submission)
    filtered_sub, invalid_tickers = validate_ids_signals(
        universe[ticker_col], submission, ticker_col
    )
    validate_values(filtered_sub, signal_col)
    return ticker_col, signal_col, filtered_sub, invalid_tickers


def validate_submission_crypto(
    universe: pd.DataFrame, submission: pd.DataFrame
) -> Tuple[str, str, pd.DataFrame, List[str]]:
    """Validate the headers, ids, and values for a submission.

    Arguments:
        universe: pd.DataFrame - the live universe of ids on which the predictions are based
        submission: pd.DataFrame - the predictions to validate

    Returns:
        Tuple[str, str, pd.DataFrame, List[str]] - the validated ticker column, signal column,
                                                   filtered submission, and list of invalid tickers
    """
    print(universe)
    ticker_col, signal_col = validate_headers_crypto(submission)
    filtered_sub, invalid_tickers = validate_ids_crypto(
        universe[ticker_col], submission, ticker_col
    )
    validate_values(filtered_sub, signal_col)
    return ticker_col, signal_col, filtered_sub, invalid_tickers


def remap_ids(
    data: pd.DataFrame,
    ticker_map: pd.Series | pd.DataFrame,
    src_id_col: str,
    dst_id_col: str,
) -> pd.DataFrame:
    """Join the data to the ticker map based on source ids
    and remap to the destination ids. If the ticker is a Series, it is assumed that
    src_id_col and dst_id_col are the same, and the ticker map is simply used to
    ensure the data has all ids in the ticker map.

    Arguments:
        data: pd.DataFrame - the data to remap
        ticker_map: pd.Series | pd.DataFrame - the mapping of source ids to destination ids
        src_id_col: str - the name of the source ids column in the data
        dst_id_col: str - the name of the destination ids column in the ticker map
    """
    # first, index the universe and data on the source ids
    indexed_map = ticker_map.reset_index().set_index(src_id_col)
    indexed_data = data.set_index(src_id_col)
    return (
        # then, join the universe and data
        indexed_map.join(indexed_data)
        # get just the destination ids and prediction columns
        .reset_index()[[dst_id_col, *indexed_data.columns]]
        # finally, sort by the destination ticker column
        .sort_values(dst_id_col)
    )


def clean_submission(
    universe: pd.Series | pd.DataFrame,
    submission: pd.DataFrame,
    src_id_col: str,
    src_signal_col: str,
    dst_id_col: Optional[str] = None,
    dst_signal_col: Optional[str] = None,
    rank_and_fill: bool = False,
) -> pd.Series:
    """Prepares your submission for uploading to a Numerai tournament.
    Joins your submission to the universe, remaps ids as neded, drops
    duplicates, sets ids as index, renames the series, then optionally
    tie-kept ranks and fills NaNs with 0.5.

    This function is used in Numerai to clean submissions for use in the
    Meta Model and scoring. We rank and fill submissions before scoring.

    Arguments:
        universe: pd.Series - the live universe of ids on which the predictions are based
        submission: pd.DataFrame - the submission to clean
        src_id_col: str - the name of the ids column
        src_signal_col: str - the name of the predictions column
        dst_id_col: Optional[str] - optional name of the id column to map the ids to
        dst_signal_col: Optional[str] - optional name of the signal column to rename the submission to
        rank_and_fill: bool - whether to call tie_kept_rank and then fill NaNs with 0.5

    Returns:
        pd.Series - the cleaned, properly indexed submission
    """
    assert len(universe) > 0, "universe must not be empty"
    if isinstance(universe, pd.DataFrame):
        assert universe.isna().sum().sum() == 0, "universe must not contain NaNs"
    else:
        assert universe.isna().sum() == 0, "universe must not contain NaNs"
    assert len(submission) > 0, "predictions must not be empty"

    if dst_id_col is None:
        dst_id_col = src_id_col
    if dst_signal_col is None:
        dst_signal_col = src_signal_col

    clean_preds = (
        remap_ids(submission, universe, src_id_col, dst_id_col)
        # drop NaNs and duplicates
        .dropna(subset=[dst_id_col])
        .drop_duplicates(subset=dst_id_col, keep="first")
        # set ids as index and sort
        .set_index(dst_id_col)
        .sort_index()
        # rename to given name
        .rename(columns={src_signal_col: dst_signal_col})
    )[dst_signal_col]
    # rank and fill with 0.5
    if rank_and_fill:
        clean_preds = tie_kept_rank(clean_preds).fillna(0.5)
    return clean_preds
