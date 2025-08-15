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
    assert valid_headers, (
        "invalid_submission_headers: headers must be one of"
        f" {expected_id_cols} and one of {expected_pred_cols}"
    )
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


def remap_ids(
    data: pd.DataFrame,
    ticker_map: pd.Series | pd.DataFrame,
    src_id_col: str,
    dst_id_col: str,
) -> pd.DataFrame:
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
    live_ids: pd.Series | pd.DataFrame,
    predictions: pd.DataFrame,
    ticker_col: str,
    signal_col: str,
    rename_as: Optional[str],
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
        ticker_col: str - the name of the ids column
        signal_col: str - the name of the predictions column
        rename_as: Optional[str] - the string to which the submission should be renamed
        id_col: str - the column name of the ids
        rank_and_fill: bool - whether to rank and fill NaNs with 0.5

    Returns:
        pd.Series - the cleaned prediction series with ids as index
    """
    assert len(live_ids) > 0, "live_ids must not be empty"
    if isinstance(live_ids, pd.DataFrame):
        assert live_ids.isna().sum().sum() == 0, "live_ids must not contain NaNs"
    else:
        assert live_ids.isna().sum() == 0, "live_ids must not contain NaNs"
    assert len(predictions) > 0, "predictions must not be empty"

    clean_preds = (
        remap_ids(predictions, live_ids, ticker_col, id_col)
        # drop NaNs and duplicates
        .dropna(subset=[id_col])
        .drop_duplicates(subset=id_col, keep="first")
        # set ids as index and sort
        .set_index(id_col)
        .sort_index()
        # rename to given name
        .rename(columns={signal_col: rename_as})
    )[rename_as]
    # rank and fill with 0.5
    if rank_and_fill:
        clean_preds = tie_kept_rank(clean_preds).fillna(0.5)
    return clean_preds


def validate_and_clean_submission_numerai(
    universe: pd.Series,
    submission: pd.DataFrame,
    id_col: str = "id",
    rename_as: Optional[str] = None,
    rank_and_fill: bool = False,
) -> pd.Series:
    ticker_col, signal_col = validate_headers_numerai(submission)
    filtered_sub, invalid_tickers = validate_ids_numerai(
        universe, submission, ticker_col
    )
    validate_values(filtered_sub, signal_col)
    return clean_submission(
        live_ids=universe,
        predictions=filtered_sub,
        ticker_col=ticker_col,
        signal_col=signal_col,
        rename_as=rename_as,
        id_col=id_col,
        rank_and_fill=rank_and_fill,
    )


def validate_and_clean_submission_signals(
    universe: pd.DataFrame,
    submission: pd.DataFrame,
    id_col: str,
    rename_as: Optional[str] = None,
    rank_and_fill: bool = True,
) -> pd.Series:
    # drop data_type and date columns if they exist
    if "data_type" in submission.columns:
        logger.warning(
            "data_type column found in Signals submission. This is deprecated and support will be removed in the future. "
            "Please remove the data_type column from your Signals submission."
        )
    date_col = [
        date_col
        for date_col in SIGNALS_ALLOWED_DATE_COLS
        if date_col in list(submission.columns)
    ]
    submission = submission.drop(columns=["data_type", *date_col], errors="ignore")
    ticker_col, signal_col = validate_headers_signals(submission)
    filtered_sub, invalid_tickers = validate_ids_signals(
        universe[ticker_col], submission, ticker_col
    )
    validate_values(filtered_sub, signal_col)
    return clean_submission(
        live_ids=universe,
        predictions=filtered_sub,
        ticker_col=ticker_col,
        signal_col=signal_col,
        rename_as=rename_as,
        id_col=id_col,
        rank_and_fill=rank_and_fill,
    )


def validate_and_clean_submission_crypto(
    universe: pd.DataFrame,
    submission: pd.DataFrame,
    id_col: str = "symbol",
    rename_as: Optional[str] = None,
    rank_and_fill: bool = True,
):
    ticker_col, signal_col = validate_headers_crypto(submission)
    filtered_sub, invalid_tickers = validate_ids_crypto(
        universe[ticker_col], submission, ticker_col
    )
    validate_values(filtered_sub, signal_col)
    return clean_submission(
        live_ids=universe,
        predictions=filtered_sub,
        ticker_col=ticker_col,
        signal_col=signal_col,
        rename_as=rename_as,
        id_col=id_col,
        rank_and_fill=rank_and_fill,
    )
