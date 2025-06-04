import unittest
import random
import string
from typing import List

import numpy as np
import pandas as pd  # type: ignore

from numerai_tools.submissions import (
    NUMERAI_ALLOWED_ID_COLS,
    NUMERAI_ALLOWED_PRED_COLS,
    SIGNALS_ALLOWED_ID_COLS,
    SIGNALS_ALLOWED_PRED_COLS,
    CRYPTO_ALLOWED_ID_COLS,
    CRYPTO_ALLOWED_PRED_COLS,
    _validate_headers,
    validate_headers_numerai,
    validate_headers_signals,
    validate_headers_crypto,
    validate_values,
    _validate_ids,
    validate_ids_numerai,
    validate_ids_signals,
    validate_ids_crypto,
    clean_predictions,
)


class TestSubmissions(unittest.TestCase):
    def setUp(self):
        # use 9 digits for cusip handling checks
        self.ids = generate_ids(9, 5)
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
        self.crypto_subs = [
            generate_submission(self.ids, id_col, pred_col)
            for id_col in CRYPTO_ALLOWED_ID_COLS
            for pred_col in CRYPTO_ALLOWED_PRED_COLS
        ]

    def test_validate_headers(self):
        assert _validate_headers(
            ["test1"], ["test2"], generate_submission(self.ids, "test1", "test2")
        ) == ("test1", "test2")

    def test_validate_headers_wrong_name(self):
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            _validate_headers,
            ["test1"],
            ["test2"],
            generate_submission(self.ids, "wrong", "test2"),
        )
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            _validate_headers,
            ["test1"],
            ["test2"],
            generate_submission(self.ids, "test1", "wrong"),
        )

    def test_validate_headers_missing(self):
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            _validate_headers,
            ["test1"],
            ["test2"],
            generate_submission(self.ids, "test1", "test2")[["test1"]],
        )
        self.assertRaisesRegex(
            AssertionError,
            "headers must be one of",
            _validate_headers,
            ["test1"],
            ["test2"],
            generate_submission(self.ids, "test1", "test2")[["test2"]],
        )

    def test_validate_headers_numerai(self):
        for sub in self.classic_subs:
            assert validate_headers_numerai(sub) == tuple(sub.columns)

    def test_validate_headers_numerai_wrong_name(self):
        for sub in self.classic_subs:
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_numerai,
                sub.rename(columns={sub.columns[0]: "wrong"}),
            )
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_numerai,
                sub.rename(columns={sub.columns[1]: "wrong"}),
            )

    def test_validate_headers_numerai_missing(self):
        for sub in self.classic_subs:
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_numerai,
                sub[[sub.columns[0]]],
            )
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_numerai,
                sub[[sub.columns[1]]],
            )

    def test_validate_headers_signals(self):
        for sub in self.signals_subs:
            assert validate_headers_signals(sub) == tuple(sub.columns)

    def test_validate_headers_signals_wrong_name(self):
        for sub in self.signals_subs:
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_signals,
                sub.rename(columns={sub.columns[0]: "wrong"}),
            )
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_signals,
                sub.rename(columns={sub.columns[1]: "wrong"}),
            )

    def test_validate_headers_signals_missing(self):
        for sub in self.signals_subs:
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_signals,
                sub[[sub.columns[0]]],
            )
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_signals,
                sub[[sub.columns[1]]],
            )

    def test_validate_headers_crypto(self):
        for sub in self.crypto_subs:
            assert validate_headers_crypto(sub) == tuple(sub.columns)

    def test_validate_headers_crypto_wrong_name(self):
        for sub in self.crypto_subs:
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_crypto,
                sub.rename(columns={sub.columns[0]: "wrong"}),
            )
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_crypto,
                sub.rename(columns={sub.columns[1]: "wrong"}),
            )

    def test_validate_headers_crypto_missing(self):
        for sub in self.crypto_subs:
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_crypto,
                sub[[sub.columns[0]]],
            )
            self.assertRaisesRegex(
                AssertionError,
                "headers must be one of",
                validate_headers_crypto,
                sub[[sub.columns[1]]],
            )

    def test_validate_values(self):
        validate_values(generate_submission(self.ids, "id", "prediction"), "prediction")

    def test_validate_values_nans(self):
        nan_sub = generate_submission(self.ids, "id", "prediction")
        nan_sub.loc[0, "prediction"] = np.nan
        self.assertRaisesRegex(
            AssertionError,
            "must not contain NaNs",
            validate_values,
            nan_sub,
            "prediction",
        )

    def test_validate_values_out_of_bounds(self):
        out_of_bounds_sub = generate_submission(self.ids, "id", "prediction")
        out_of_bounds_sub.loc[0, "prediction"] = -1
        self.assertRaisesRegex(
            AssertionError,
            "values must be between 0 and 1 exclusive",
            validate_values,
            out_of_bounds_sub,
            "prediction",
        )
        out_of_bounds_sub.loc[0, "prediction"] = 2
        self.assertRaisesRegex(
            AssertionError,
            "values must be between 0 and 1 exclusive",
            validate_values,
            out_of_bounds_sub,
            "prediction",
        )

    def test_validate_values_zero_std(self):
        const_sub = generate_submission(self.ids, "id", "prediction")
        const_sub["prediction"] = 0.5
        self.assertRaisesRegex(
            AssertionError,
            "submission must have non-zero standard deviation",
            validate_values,
            const_sub,
            "prediction",
        )

    def test_validate_ids(self):
        sub = generate_submission(self.ids, "id", "prediction")
        new_sub, invalid_ids = _validate_ids(self.ids, sub, "id", len(self.ids))
        assert (new_sub == sub.sort_values("id")).all().all()
        assert invalid_ids == []

    def test_validate_ids_nans(self):
        nan_sub = generate_submission(self.ids, "id", "prediction")
        nan_sub.loc[0, "id"] = np.nan
        self.assertRaisesRegex(
            AssertionError,
            "must not contain NaNs",
            _validate_ids,
            self.ids,
            nan_sub,
            "id",
            len(self.ids),
        )

    def test_validate_ids_all_nan_ids(self):
        nan_ids = pd.Series([np.nan, np.nan, np.nan])
        submission = generate_submission(nan_ids, "id", "prediction")
        self.assertRaisesRegex(
            AssertionError,
            "Submission must not contain NaNs",
            _validate_ids,
            self.ids,
            submission,
            "id",
            len(self.ids),
        )

    def test_validate_ids_duplicates(self):
        dup_sub = generate_submission(self.ids, "id", "prediction")
        dup_sub.loc[0] = dup_sub.loc[1]
        self.assertRaisesRegex(
            AssertionError,
            "Duplicates detected",
            _validate_ids,
            self.ids,
            dup_sub,
            "id",
            len(self.ids),
        )

    def test_validate_ids_duplicate_ids(self):
        submission = generate_submission(self.ids, "id", "prediction")
        submission = pd.concat([submission, submission.iloc[:1]])
        self.assertRaisesRegex(
            AssertionError,
            "Duplicates detected",
            _validate_ids,
            self.ids,
            submission,
            "id",
            len(self.ids),
        )

    def test_validate_ids_missing(self):
        missing_sub = generate_submission(self.ids, "id", "prediction")
        missing_sub = missing_sub[missing_sub["id"] != self.ids[0]]
        self.assertRaisesRegex(
            AssertionError,
            "Not enough stocks submitted",
            _validate_ids,
            self.ids,
            missing_sub,
            "id",
            len(self.ids),
        )

    def test_validate_ids_empty_submission(self):
        empty_submission = pd.DataFrame(columns=["id", "prediction"])
        self.assertRaisesRegex(
            AssertionError,
            "Not enough stocks submitted.",
            _validate_ids,
            self.ids,
            empty_submission,
            "id",
            len(self.ids),
        )

    def test_validate_ids_all_invalid_ids(self):
        invalid_ids = pd.Series(["invalid1", "invalid2", "invalid3"])
        submission = generate_submission(invalid_ids, "id", "prediction")
        self.assertRaisesRegex(
            AssertionError,
            "Not enough stocks submitted.",
            _validate_ids,
            self.ids,
            submission,
            "id",
            len(self.ids),
        )

    def test_validate_ids_mixed_valid_invalid_ids(self):
        mixed_ids = self.ids.tolist() + ["invalid1", "invalid2"]
        submission = generate_submission(mixed_ids, "id", "prediction")
        new_sub, invalid_ids = _validate_ids(self.ids, submission, "id", len(self.ids))
        assert (new_sub["id"] == self.ids.sort_values()).all()
        assert set(invalid_ids) == {"invalid1", "invalid2"}

    def test_validate_ids_numerai(self):
        sub = generate_submission(self.ids, "id", "prediction")
        new_sub, invalid_ids = validate_ids_numerai(self.ids, sub, "id")
        assert (new_sub == sub.sort_values("id")).all().all()
        assert invalid_ids == []

    def test_validate_ids_signals(self):
        ids = generate_ids(9, 100)
        sub = generate_submission(ids, "ticker", "signal")
        new_sub, invalid_ids = validate_ids_signals(ids, sub, "ticker")
        assert (new_sub == sub.sort_values("ticker")).all().all()
        assert invalid_ids == []

    def test_validate_ids_crypto(self):
        ids = generate_ids(9, 100)
        sub = generate_submission(ids, "ticker", "signal")
        new_sub, invalid_ids = validate_ids_crypto(ids, sub, "ticker")
        assert (new_sub == sub.sort_values("ticker")).all().all()
        assert invalid_ids == []

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

    def test_clean_predictions_rank_and_fill(self):
        int_sub = generate_submission(self.ids, "id", "prediction", random_vals=False)
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

    def test_clean_predictions_empty_predictions(self):
        empty_predictions = pd.DataFrame(columns=["id", "prediction"])
        self.assertRaisesRegex(
            AssertionError,
            "predictions must not be empty",
            clean_predictions,
            self.ids,
            empty_predictions,
            id_col="id",
            rank_and_fill=False,
        )

    def test_clean_predictions_all_nan_predictions(self):
        predictions = generate_submission(self.ids, "id", "prediction")
        predictions["prediction"] = np.nan
        cleaned_predictions = clean_predictions(
            self.ids,
            predictions,
            id_col="id",
            rank_and_fill=True,
        )
        assert (cleaned_predictions == 0.5).all().all()

    def test_clean_predictions_mixed_valid_invalid_ids(self):
        mixed_ids = self.ids.tolist() + ["invalid1", "invalid2"]
        predictions = generate_submission(mixed_ids, "id", "prediction")
        cleaned_predictions = clean_predictions(
            self.ids,
            predictions,
            id_col="id",
            rank_and_fill=False,
        )
        assert (cleaned_predictions.index == self.ids.sort_values()).all()

    def test_clean_predictions_duplicate_ids(self):
        predictions = generate_submission(self.ids, "id", "prediction")
        predictions = pd.concat([predictions, predictions.iloc[:1]])
        cleaned_predictions = clean_predictions(
            self.ids,
            predictions,
            id_col="id",
            rank_and_fill=False,
        )
        assert not cleaned_predictions.index.duplicated().any()


def generate_ids(id_length: int, num_rows: int) -> List[str]:
    """Generates a given number of unique ascii-valued strings of a given length.

    Arguments:
        id_length -- integer length of the id
        num_rows -- integer number of rows to generate

    Return List[str]:
        - list of unique ascii-valued strings of the given
    """
    values: set[str] = set()
    while len(values) < num_rows:
        new_value = "".join(random.choices(string.ascii_uppercase, k=id_length))
        values.add(new_value)
    return pd.Series(list(values))


def generate_submission(
    live_ids: List[str],
    id_col: str,
    pred_col: str,
    random_vals: bool = True,
    legacy_headers: dict = {},
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


if __name__ == "__main__":
    unittest.main()
