from numerai_tools.scoring import tie_kept_rank

from typing import Tuple, List

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
SIGNALS_MIN_TICKERS = 100

CRYPTO_ALLOWED_ID_COLS = ["symbol"]
CRYPTO_ALLOWED_PRED_COLS = ["prediction", "signal"]
CRYPTO_MIN_TICKERS = 100


def _validate_headers(
    expected_id_cols: List[str], expected_pred_cols: List[str], submission: pd.DataFrame
) -> Tuple[str, str]:
    """Validate the given submission has the right headers.
    It is recommended to use one of the following functions instead of this one:
        - validate_headers_numerai
        - validate_headers_signals

    Arguments:
        submission -- pandas DataFrame of the submission

    Return Tuple[str, str]:
        - string name of the id column
        - string name of the prediction column
    """
    expected_headers = [
        [ticker_col, signal_col]
        for ticker_col in expected_id_cols
        for signal_col in expected_pred_cols
    ]
    columns = submission.columns
    valid_headers = list(columns) in expected_headers
    assert (
        valid_headers
    ), f"headers must be one of {expected_id_cols} and one of {expected_pred_cols}"
    return columns[0], columns[1]


def validate_headers_numerai(submission: pd.DataFrame) -> Tuple[str, str]:
    return _validate_headers(
        NUMERAI_ALLOWED_ID_COLS, NUMERAI_ALLOWED_PRED_COLS, submission
    )


def validate_headers_signals(submission: pd.DataFrame) -> Tuple[str, str]:
    return _validate_headers(
        SIGNALS_ALLOWED_ID_COLS, SIGNALS_ALLOWED_PRED_COLS, submission
    )


def validate_headers_crypto(submission: pd.DataFrame) -> Tuple[str, str]:
    return _validate_headers(
        CRYPTO_ALLOWED_ID_COLS, CRYPTO_ALLOWED_PRED_COLS, submission
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
    ), "submission must not contain NaNs"
    assert (
        submission[prediction_col].between(0, 1).all()
    ), "values must be between 0 and 1 exclusive"
    assert not np.isclose(
        0, submission[prediction_col].std()
    ), "submission must have non-zero standard deviation"


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
    ), f"Submission must not contain NaNs in the {id_col} column."

    index_sub = submission.copy()
    index_sub[id_col] = index_sub[id_col].astype(str)

    live_ids = live_ids.astype(str)
    live_sub = index_sub[index_sub[id_col].isin(live_ids)].sort_values(id_col)
    assert (
        not live_sub[id_col].duplicated().any()
    ), f"Duplicates detected in {id_col} for live period."

    # join on live_ids and ensure min tickers reached
    assert (
        len(live_sub) >= min_tickers
    ), f"Not enough stocks submitted. Are you using the latest live ids or live universe?"

    invalid_tickers = list(set(index_sub[id_col]).difference(set(live_sub[id_col])))
    return live_sub, invalid_tickers


def validate_ids_numerai(live_ids: pd.Series, submission: pd.DataFrame, id_col: str):
    return _validate_ids(live_ids, submission, id_col, len(live_ids))


def validate_ids_signals(live_ids: pd.Series, submission: pd.DataFrame, id_col: str):
    return _validate_ids(live_ids, submission, id_col, SIGNALS_MIN_TICKERS)


def validate_ids_crypto(live_ids: pd.Series, submission: pd.DataFrame, id_col: str):
    return _validate_ids(live_ids, submission, id_col, CRYPTO_MIN_TICKERS)


def clean_predictions(
    live_ids: pd.Series,
    predictions: pd.DataFrame,
    id_col: str,
    rank_and_fill: bool,
) -> pd.Series:
    """Prepare predictions for submission to Numerai.
    Filters out ids not in live data, drops duplicates, sets ids as index,
    then optionally ranks (keeping ties) and fills NaNs with 0.5.

    This function is used in Numerai to clean submissions for use in the
    Meta Model and scoring. We only rank and fill in preparation for scoring
    Signals and Crypto submissions.

    Arguments:
        live_ids: pd.Series - the ids in the live data
        predictions: pd.DataFrame - the predictions to clean
        id_col: str - the column name of the ids
        rank_and_fill: bool - whether to rank and fill NaNs with 0.5
    """
    assert len(live_ids) > 0, "live_ids must not be empty"
    assert live_ids.isna().sum() == 0, "live_ids must not contain NaNs"
    assert len(predictions) > 0, "predictions must not be empty"

    # drop null indices
    predictions = predictions[~predictions[id_col].isna()]
    predictions = (
        predictions
        # filter out ids not in live data
        [predictions[id_col].isin(live_ids)]
        # drop duplicate ids (keep first)
        .drop_duplicates(subset=id_col, keep="first")
        # set ids as index
        .set_index(id_col).sort_index()
    )
    # rank and fill with 0.5
    if rank_and_fill:
        predictions = tie_kept_rank(predictions).fillna(0.5)
    return predictions
