import unittest

import numpy as np
import pandas as pd  # type: ignore

from numerai_tools.scoring import (
    numerai_corr,
    alpha,
    meta_portfolio_contribution,
)


class TestScoring(unittest.TestCase):
    def setUp(self):
        self.up = pd.Series(list(range(5))).rename("up")
        self.down = pd.Series(list(reversed(range(5)))).rename("down")
        self.up_down = pd.Series([1, 0, 1, 0, 1]).rename("up_down")
        self.down_up = (1 - self.up_down).rename("down_up")
        self.up_float = (self.up / self.up.max()).rename("up_float")
        self.pos_neg = pd.Series([0, -0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]).rename(
            "pos_neg"
        )

    def test_numerai_corr_doesnt_clobber_targets(self):
        s = [x / 100 for x in range(100)]
        df = pd.DataFrame({"target": s, "prediction": s})
        numerai_corr(df[["prediction"]], df["target"])
        assert pd.Series(s).equals(df["target"]), f"{s} != {list(df['target'].values)}"

    def test_numerai_corr_target_pow15_option(self):
        # ensure the target_pow15 argument to numerai_corr operates correctly
        s = [x / 100 for x in range(100)]
        df = pd.DataFrame({"target": s, "prediction": s})
        corr_w_pow = numerai_corr(df[["prediction"]], df["target"], target_pow15=True)
        corr_wo_pow = numerai_corr(df[["prediction"]], df["target"], target_pow15=False)
        # we would expect the correlation to be higher when using the pow15 transformation
        # since the predictions are rank-gauss-pow1.5 transformed in numerai_corr
        assert abs(corr_w_pow[0]) > abs(corr_wo_pow[0])

    def test_alpha(self):
        s = pd.DataFrame([[1, 2, 3, 4, 5]]).T
        N = pd.DataFrame(
            [
                [1, 5],
                [2, 4],
                [3, 3],
                [4, 2],
                [5, 1],
            ]
        )
        v = pd.Series([1, 0.5, 1, 0.5, 1]).T
        t = pd.Series([1, 0, 1, 0, 1]).T
        score = alpha(s, N, v, t)
        np.testing.assert_allclose(score, 0.0, atol=1e-14, rtol=1e-14)

    def test_meta_portfolio_contribution(self):
        s = pd.DataFrame([[1, 2, 3, 4, 5], [1, 2, 1, 2, 1]]).T
        st = pd.Series([0.6, 0.4])
        N = pd.DataFrame(
            [
                [1, 5],
                [2, 4],
                [3, 3],
                [4, 2],
                [5, 1],
            ]
        )
        v = pd.Series([3, 2, 1, 2, 3]).T
        t = pd.Series([1.0, 2.0, 3.0, 2.0, 1.0]).T
        score = meta_portfolio_contribution(s, st, N, v, t)
        assert np.isclose(score[0], -0.04329786867021718)
        assert np.isclose(score[1], 0.06494680300532589)


if __name__ == "__main__":
    unittest.main()
