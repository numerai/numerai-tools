import unittest

import numpy as np
import pandas as pd  # type: ignore

from numerai_tools.signals import (
    churn,
    turnover,
    calculate_max_churn_and_turnover,
)
from .util import (
    generate_fake_universe,
    generate_new_submission,
)


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

    def test_turnover(self):
        assert np.isclose(turnover(self.up, self.up), 0)
        assert np.isclose(turnover(self.up, self.up_down), 3)
        assert np.isclose(turnover(self.up, self.oscillate), 4.5)
        assert np.isclose(turnover(self.up, self.down), 6)
        assert np.isclose(turnover(self.up, self.constant), 3.5)

    def test_churn_first_submission(self):
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
            curr_sub=fake_submission,
            curr_neutralizer=fake_neutralizers,
            curr_weight=fake_sample_weights,
            prev_week_subs=[],
            prev_neutralizers={"20240208": fake_neutralizers},
            prev_sample_weights={"20240208": fake_sample_weights},
            universe=fake_universe.set_index("numerai_ticker").sort_index(),
            curr_signal_col="signal",
            curr_ticker_col="numerai_ticker",
        )
        assert np.isclose(churn, 1)
        assert np.isclose(turnover, 1)

    def test_churn_handles_different_id_columns(self):
        """
        Test that the churn function works when
        previous submission has different id columns.
        """
        fake_universe = generate_fake_universe("20130308")
        fake_submission = generate_new_submission(fake_universe, legacy_headers=True)
        new_fake_universe = generate_fake_universe(
            date_value="20130308", ticker_col="ticker"
        )
        fake_universe["ticker"] = new_fake_universe["ticker"]
        prev_submission = fake_submission.copy()
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
        # switch out the numerai_ticke col in-place
        prev_submission["numerai_ticker"] = new_fake_universe["ticker"]
        prev_submission.rename(columns={"numerai_ticker": "ticker"}, inplace=True)
        prev_neutralizers = fake_neutralizers.copy()
        prev_neutralizers.index = new_fake_universe["ticker"]
        prev_neutralizers.index.name = "ticker"
        prev_sample_weights = fake_sample_weights.copy()
        prev_sample_weights.index = new_fake_universe["ticker"]
        prev_sample_weights.index.name = "ticker"
        churn, turnover = calculate_max_churn_and_turnover(
            curr_sub=fake_submission,
            curr_neutralizer=fake_neutralizers,
            curr_weight=fake_sample_weights,
            prev_week_subs={"20240208": prev_submission},
            prev_neutralizers={"20240208": prev_neutralizers},
            prev_sample_weights={"20240208": prev_sample_weights},
            universe=fake_universe.set_index("numerai_ticker").sort_index(),
            curr_signal_col="signal",
            curr_ticker_col="numerai_ticker",
        )
        assert np.isclose(churn, 0)
        assert np.isclose(turnover, 0)


if __name__ == "__main__":
    unittest.main()
