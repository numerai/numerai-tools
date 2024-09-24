import unittest

import numpy as np
import pandas as pd  # type: ignore

from numerai_tools.signals import churn


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
