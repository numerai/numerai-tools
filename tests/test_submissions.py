import unittest
import random
import string
from typing import List, Optional

import numpy as np
import pandas as pd

from numerai_tools.submissions import (
    NUMERAI_ALLOWED_ID_COLS,
    NUMERAI_ALLOWED_PRED_COLS,
    SIGNALS_ALLOWED_ID_COLS,
    SIGNALS_ALLOWED_PRED_COLS,
    validate_headers,
    validate_headers_numerai,
    validate_headers_signals,
    validate_values,
    validate_ids,
    clean_predictions,
)


class TestSubmissions(unittest.TestCase):
    def setUp(self):
        # use 9 digits for cusip handling checks
        self.ids = pd.Series(generate_ids(9, 5))
        self.classic_subs = [
            generate_submission(self.ids, id_col, pred_col)
            for id_col in NUMERAI_ALLOWED_ID_COLS
            for pred_col in NUMERAI_ALLOWED_PRED_COLS
        ]
        self.signals_subs = [
            generate_submission(self.ids, id_col, pred_col)
            for id_col in SIGNALS_ALLOWED_ID_COLS
            for pred_col in SIGNALS_ALLOWED_PRED_COLS
        ]

    def test_validate_headers(self):
        for sub in self.classic_subs:
            assert validate_headers(
                NUMERAI_ALLOWED_ID_COLS, NUMERAI_ALLOWED_PRED_COLS, sub
            ) == tuple(sub.columns)
        for sub in self.signals_subs:
            assert validate_headers(
                SIGNALS_ALLOWED_ID_COLS, SIGNALS_ALLOWED_PRED_COLS, sub
            ) == tuple(sub.columns)
        bad_headers = generate_submission(self.ids, "test1", "test2")
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            validate_headers,
            NUMERAI_ALLOWED_ID_COLS,
            NUMERAI_ALLOWED_PRED_COLS,
            bad_headers,
        )
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            validate_headers,
            SIGNALS_ALLOWED_ID_COLS,
            SIGNALS_ALLOWED_PRED_COLS,
            bad_headers,
        )
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            validate_headers,
            NUMERAI_ALLOWED_ID_COLS,
            NUMERAI_ALLOWED_PRED_COLS,
            bad_headers[["test1"]],
        )
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            validate_headers,
            SIGNALS_ALLOWED_ID_COLS,
            SIGNALS_ALLOWED_PRED_COLS,
            bad_headers[["test1"]],
        )

    def test_validate_headers_numerai(self):
        for sub in self.classic_subs:
            assert validate_headers_numerai(sub) == tuple(sub.columns)
        bad_headers = generate_submission(self.ids, "test1", "test2")
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            validate_headers_numerai,
            bad_headers,
        )

    def test_validate_headers_signals(self):
        for sub in self.signals_subs:
            assert validate_headers_signals(sub) == tuple(sub.columns)
        bad_headers = generate_submission(self.ids, "test1", "test2")
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            validate_headers_signals,
            bad_headers,
        )

    def test_validate_values(self):
        ids = generate_ids(8, 10)
        classic_sub = generate_submission(ids, "id", "prediction")
        assert validate_values(classic_sub, "prediction") is None

        nan_sub = classic_sub.copy()
        nan_sub.loc[0, "prediction"] = np.nan
        self.assertRaisesRegex(
            AssertionError,
            "must not contain NaNs",
            validate_values,
            nan_sub,
            "prediction",
        )

        negative_sub = classic_sub.copy()
        negative_sub["prediction"] = -1
        self.assertRaisesRegex(
            AssertionError,
            "values must be between 0 and 1 exclusive",
            validate_values,
            negative_sub,
            "prediction",
        )

        const_sub = classic_sub.copy()
        const_sub["prediction"] = 0
        self.assertRaisesRegex(
            AssertionError,
            "submission must have non-zero standard deviation",
            validate_values,
            const_sub,
            "prediction",
        )

    def test_validate_ids(self):
        sub = self.signals_subs[0]
        id_col, pred_col = validate_headers_signals(sub)
        new_sub, invalid_ids = validate_ids(self.ids, sub, id_col, len(self.ids))
        assert (new_sub == sub.sort_values(id_col)).all().all()
        assert invalid_ids == []

        # test nans
        nan_sub = sub.copy()
        nan_sub.loc[0, id_col] = np.nan
        self.assertRaisesRegex(
            AssertionError,
            "must not contain NaNs",
            validate_ids,
            self.ids,
            nan_sub,
            id_col,
            len(self.ids),
        )

        if id_col == "cusip":
            # check that cusips are zfilled
            cusip_sub = sub.copy()
            cusip_sub.loc[0, id_col] = cusip_sub.loc[0][id_col][1:]
            cusip_ids = self.ids.copy()
            cusip_ids.loc[0] = "0" + cusip_ids[0][1:]
            new_sub, invalid_ids = validate_ids(
                cusip_ids, cusip_sub, id_col, len(self.ids)
            )
            assert (
                (new_sub[pred_col].sort_values() == cusip_sub[pred_col].sort_values())
                .all()
                .all()
            )

        # check duplicates
        dup_sub = sub.copy()
        dup_sub.loc[0] = sub.loc[1]
        self.assertRaisesRegex(
            AssertionError,
            "Duplicates detected",
            validate_ids,
            self.ids,
            dup_sub,
            id_col,
            len(self.ids),
        )

        # check missing ids
        missing_sub = sub.copy()
        missing_sub = missing_sub[missing_sub[id_col] != self.ids[0]]
        self.assertRaisesRegex(
            AssertionError,
            "Not enough stocks submitted",
            validate_ids,
            self.ids,
            missing_sub,
            id_col,
            len(self.ids),
        )

    def test_clean_predictions(self):
        int_sub = generate_submission(self.ids, "id", "prediction", random_vals=False)
        assert (
            (
                clean_predictions(
                    self.ids,
                    int_sub,
                    id_col="id",
                    rank_and_fill=False,
                ).reset_index()
                == int_sub.set_index("id").sort_index().reset_index()
            )
            .all()
            .all()
        )
        assert np.isclose(
            clean_predictions(
                self.ids,
                int_sub,
                id_col="id",
                rank_and_fill=True,
            )
            .sort_values("prediction")
            .values.T,
            [[0.1, 0.3, 0.5, 0.7, 0.9]],
        ).all()


def generate_ids(id_length: int, num_rows: int) -> List[str]:
    """Generates a given number of unique ascii-valued strings of a given length.

    Arguments:
        id_length -- integer length of the id
        num_rows -- integer number of rows to generate

    Return List[str]:
        - list of unique ascii-valued strings of the given
    """
    values = set()
    while len(values) < num_rows:
        new_value = "".join(random.choices(string.ascii_uppercase, k=id_length))
        values.add(new_value)
    return list(values)


def generate_submission(
    live_ids: List[str],
    id_col: str,
    pred_col: str,
    random_vals: bool = True,
    legacy_headers: Optional[dict] = {},
) -> pd.DataFrame:
    """Generates a random vector with given columns and ids.

    Arguments:
        live_ids -- list of strings of ids
        id_col -- string name of the id column
        pred_col -- string name of the prediction column
        random -- boolean whether to generate random values or sequential
        legacy_headers -- dictionary of legacy headers to add to the submission

    Return pd.DataFrame:
        - submission DataFrame with the given columns and ids
    """
    # if legacy_headers and date_col is None:
    #     date_col = "friday_date"
    # elif date_col is None:
    #     date_col = date_col
    # else:
    #     date_col = "date"
    rows = []
    for i, ticker in enumerate(live_ids):
        if random_vals:
            val = random.random()
        else:
            val = i
        row = {id_col: ticker, pred_col: val}
        for col, value in legacy_headers.items():
            row[col] = value
        rows.append(row)
    sub = pd.DataFrame(rows)
    return sub
