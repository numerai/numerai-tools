from numerai_tools.scoring import tie_kept_rank

import pandas as pd

NUMERAI_ALLOWED_ID_COLS = ["id"]
NUMERAI_ALLOWED_PRED_COLS = ["prediction", "probability"]

SIGNALS_ALLOWED_ID_COLS = [
    "ticker",
    "cusip",
    "sedol",
    "bloomberg_ticker",
    "composite_figi",
    "numerai_ticker",
]
SIGNALS_ALLOWED_PRED_COLS = ["prediction", "signal"]


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
    # drop null indices
    predictions = predictions[~predictions[id_col].isna()]
    predictions = (
        predictions
        # filter out ids not in live data
        [predictions[id_col].isin(live_ids)]
        # drop duplicate ids (keep first)
        .drop_duplicates(subset=id_col, keep='first')
        # set ids as index
        .set_index(id_col)
    )
    # rank and fill with 0.5
    if rank_and_fill:
        predictions = tie_kept_rank(predictions).fillna(0.5)
    return predictions
