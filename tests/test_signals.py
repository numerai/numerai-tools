import unittest
import random
import string
from typing import Callable, Optional
from unittest.mock import patch

import numpy as np
import pandas as pd

from numerai_tools.signals import (
    calculate_mean_neutral_churn,
    churn,
    neutral_churn,
    neutral_churn_penalty,
    turnover,
    calculate_max_churn_and_turnover,
)
from numerai_tools.scoring import gaussian, neutralize, tie_kept_rank


def generate_unique_values(generator: Callable, length: int, num_rows: int) -> list:
    """Generates a list of unique values using the provided generator function."""
    values: set[str] = set()
    while len(values) < num_rows:
        new_value = generator(length)
        values.add(new_value)
    return list(values)


def generate_ticker_ascii_uppercase(length: int) -> str:
    return "".join(random.choices(string.ascii_uppercase, k=length))


def generate_fake_universe(
    date_value: str = "20130308", ticker_col: str = "numerai_ticker"
) -> pd.DataFrame:
    num_rows = 100
    data = {
        "date": [date_value for _ in range(num_rows)],
        ticker_col: [
            ticker + " US"
            for ticker in generate_unique_values(
                generate_ticker_ascii_uppercase, 3, num_rows
            )
        ],
    }

    uni = pd.DataFrame(data)
    return uni


def generate_new_submission(
    universe: pd.DataFrame,
    date_value: str = "2013-03-08",
    ticker_col: str = "numerai_ticker",
    legacy_headers: bool = False,
    date_col: Optional[str] = None,
) -> pd.DataFrame:
    if legacy_headers and date_col is None:
        date_col = "friday_date"
    elif date_col is None:
        date_col = date_col
    else:
        date_col = "date"

    rows = []
    for ticker in universe[ticker_col].unique():
        if legacy_headers:
            rows.append(
                {
                    ticker_col: ticker,
                    "signal": random.random(),
                    "data_type": "live",
                    date_col: date_value,
                }
            )
        else:
            rows.append({ticker_col: ticker, "signal": random.random()})
    return pd.DataFrame(rows)


class TestSignals(unittest.TestCase):
    def setUp(self):
        self.up = pd.Series(list(range(5))).rename("up")
        self.down = pd.Series(list(reversed(range(5)))).rename("down")
        self.up_down = pd.Series([0, 1, 2, 1, 0]).rename("up_down")
        self.oscillate = pd.Series([1, 0, 1, 0, 1]).rename("oscillate")
        self.constant = pd.Series([1, 1, 1, 1, 1]).rename("pos_neg")

    def test_churn(self):
        assert np.isclose(churn(self.up, self.up), 0)
        assert np.isclose(churn(self.up, self.up_down), 1)
        assert np.isclose(churn(self.up, self.oscillate), 1)
        assert np.isclose(churn(self.up, self.down), 2)
        self.assertRaisesRegex(
            AssertionError,
            "s2 must have non-zero standard deviation",
            churn,
            self.up,
            self.constant,
        )

    def test_churn_tb(self):
        tmp = churn(self.up, self.up, top_bottom=2)
        assert np.isclose(tmp, 0), tmp
        tmp = churn(self.up, self.up_down, top_bottom=2)
        assert np.isclose(tmp, 0.5), tmp
        tmp = churn(self.up, self.oscillate, top_bottom=2)
        assert np.isclose(tmp, 0.5), tmp
        tmp = churn(self.up, self.down, top_bottom=2)
        assert np.isclose(tmp, 1), tmp
        tmp = churn(self.up, self.constant, top_bottom=2)
        assert np.isclose(tmp, 0), tmp

    def test_neutral_churn(self):
        rng = np.random.default_rng(0)
        index = [f"id{i:03d}" for i in range(100)]
        neutralizers1 = pd.DataFrame(
            rng.normal(size=(100, 3)), index=index, columns=["f0", "f1", "f2"]
        )
        neutralizers2 = pd.DataFrame(
            rng.normal(size=(100, 3)), index=index, columns=["f0", "f1", "f2"]
        )
        common_signal = rng.normal(size=100)
        s1 = pd.Series(
            common_signal + neutralizers1["f0"], index=index, name="prediction"
        )
        s2 = pd.Series(
            common_signal + neutralizers2["f0"], index=index, name="prediction"
        )

        neutral_s1 = neutralize(
            gaussian(tie_kept_rank(s1.to_frame())), neutralizers1
        ).iloc[:, 0]
        neutral_s2 = neutralize(
            gaussian(tie_kept_rank(s2.to_frame())), neutralizers2
        ).iloc[:, 0]

        neutral_churn_value = neutral_churn(s1, s2, neutralizers1, neutralizers2)
        assert np.isclose(
            neutral_churn_value,
            churn(neutral_s1, neutral_s2),
        )
        assert neutral_churn_value < churn(s1, s2)

    def test_neutral_churn_penalty(self):
        assert neutral_churn_penalty(0) == 1
        assert neutral_churn_penalty(0.099) == 1
        assert neutral_churn_penalty(0.1) == 1
        assert np.isclose(
            neutral_churn_penalty(0.2),
            min(1, 2 / (1 + np.exp(10 * (0.2 - 0.1)))),
        )
        assert neutral_churn_penalty(0.2) < neutral_churn_penalty(0.15)
        assert neutral_churn_penalty(2, scaling_factor=400) == 0

        self.assertRaises(AssertionError, neutral_churn_penalty, np.nan)
        self.assertRaises(AssertionError, neutral_churn_penalty, -0.01)
        self.assertRaises(AssertionError, neutral_churn_penalty, 0.2, 0.1, 0)

    def test_calculate_mean_neutral_churn(self):
        rng = np.random.default_rng(1)
        index = pd.Index([f"id{i:03d}" for i in range(100)], name="numerai_ticker")
        curr_sub = pd.Series(rng.random(size=100), index=index, name="signal")
        prev_subs = {
            "20260814": pd.Series(rng.random(size=100), index=index, name="signal"),
            "20260821": pd.Series(rng.random(size=100), index=index, name="signal"),
        }
        curr_neutralizer = pd.DataFrame(
            rng.normal(size=(100, 3)), index=index, columns=["f0", "f1", "f2"]
        )
        prev_neutralizers = {
            datestamp: pd.DataFrame(
                rng.normal(size=(100, 3)),
                index=index,
                columns=["f0", "f1", "f2"],
            )
            for datestamp in prev_subs
        }
        sample_weight = pd.Series(1.0, index=index, name="sample_weight")
        prev_sample_weights = {
            datestamp: sample_weight.copy() for datestamp in prev_subs
        }

        ranked_curr_sub = tie_kept_rank(curr_sub)
        expected_churns = [
            neutral_churn(
                ranked_curr_sub,
                tie_kept_rank(prev_sub),
                curr_neutralizer,
                prev_neutralizers[datestamp],
            )
            for datestamp, prev_sub in prev_subs.items()
        ]

        assert np.isclose(
            calculate_mean_neutral_churn(
                curr_sub,
                curr_neutralizer,
                sample_weight,
                prev_subs,
                prev_neutralizers,
                prev_sample_weights,
            ),
            np.mean(expected_churns),
        )
        assert (
            calculate_mean_neutral_churn(
                curr_sub,
                curr_neutralizer,
                sample_weight,
                {},
                {},
                {},
            )
            == 1
        )

        wrong_index = pd.Index(
            [f"wrong{i:03d}" for i in range(100)], name="numerai_ticker"
        )
        with self.assertRaisesRegex(
            AssertionError,
            "does not have enough overlapping ids",
        ):
            calculate_mean_neutral_churn(
                curr_sub,
                curr_neutralizer.set_axis(wrong_index),
                sample_weight,
                prev_subs,
                prev_neutralizers,
                prev_sample_weights,
            )

        with patch(
            "numerai_tools.signals.churn",
            side_effect=AssertionError("s2 must have non-zero standard deviation"),
        ):
            assert (
                calculate_mean_neutral_churn(
                    curr_sub,
                    curr_neutralizer,
                    sample_weight,
                    prev_subs,
                    prev_neutralizers,
                    prev_sample_weights,
                )
                == 1
            )

    def test_turnover(self):
        assert np.isclose(turnover(self.up, self.up), 0)
        assert np.isclose(turnover(self.up, self.up_down), 3)
        assert np.isclose(turnover(self.up, self.oscillate), 4.5)
        assert np.isclose(turnover(self.up, self.down), 6)
        assert np.isclose(turnover(self.up, self.constant), 3.5)

    def test_churn_and_turnover_first_submission(self):
        """
        Test that the churn function works for the first submission
        No exceptions should be raised, should return 1
        """
        fake_universe = generate_fake_universe("20130308")
        fake_submission = generate_new_submission(fake_universe)
        fake_neutralizers = pd.DataFrame(
            {
                "neutralizer_1": [0.1] * len(fake_universe),
                "neutralizer_2": [0.2] * len(fake_universe),
            },
            index=fake_universe["numerai_ticker"],
        )
        fake_sample_weights = pd.Series(
            [0.5] * len(fake_universe),
            index=fake_universe["numerai_ticker"],
            name="sample_weight",
        )
        churn, turnover = calculate_max_churn_and_turnover(
            curr_sub=fake_submission.set_index("numerai_ticker"),
            curr_neutralizer=fake_neutralizers,
            curr_sample_weight=fake_sample_weights,
            prev_subs={},
            prev_neutralizers={},
            prev_sample_weights={},
        )
        assert np.isclose(churn, 1)
        assert np.isclose(turnover, 1)

    def test_churn_and_turnover_same_submission(self):
        """
        Test that the churn function works when
        previous submission has different id columns.
        """
        fake_universe = generate_fake_universe("20240209")
        fake_submission = generate_new_submission(fake_universe)
        fake_submission = fake_submission.set_index("numerai_ticker")
        fake_neutralizers = pd.DataFrame(
            {
                "neutralizer_1": [0.1] * len(fake_universe),
                "neutralizer_2": [0.2] * len(fake_universe),
            },
            index=fake_universe["numerai_ticker"],
        )
        fake_sample_weights = pd.Series(
            [0.5] * len(fake_universe),
            index=fake_universe["numerai_ticker"],
            name="sample_weight",
        )
        churn, turnover = calculate_max_churn_and_turnover(
            curr_sub=fake_submission,
            curr_neutralizer=fake_neutralizers,
            curr_sample_weight=fake_sample_weights,
            prev_subs={"20240208": fake_submission.copy()},
            prev_neutralizers={"20240208": fake_neutralizers.copy()},
            prev_sample_weights={"20240208": fake_sample_weights.copy()},
        )
        assert np.isclose(churn, 0)
        assert np.isclose(turnover, 0)

        with patch(
            "numerai_tools.signals.churn",
            side_effect=AssertionError("s1 must have non-zero standard deviation"),
        ):
            with self.assertRaisesRegex(
                AssertionError,
                "s1 must have non-zero standard deviation",
            ):
                calculate_max_churn_and_turnover(
                    curr_sub=fake_submission,
                    curr_neutralizer=fake_neutralizers,
                    curr_sample_weight=fake_sample_weights,
                    prev_subs={"20240208": fake_submission.copy()},
                    prev_neutralizers={"20240208": fake_neutralizers.copy()},
                    prev_sample_weights={"20240208": fake_sample_weights.copy()},
                )


if __name__ == "__main__":
    unittest.main()
