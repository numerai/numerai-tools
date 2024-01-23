import unittest

import numpy as np
import pandas as pd

from numerai_tools.scoring import (
    correlation,
    numerai_corr,
    tie_broken_rank_correlation,
    spearman_correlation,
    pearson_correlation,
    tie_broken_rank,
    tie_kept_rank,
    gaussian,
    neutralize,
    one_hot_encode,
    power,
    tie_kept_rank__gaussianize__pow_1_5,
    variance_normalize,
    orthogonalize,
    stake_weight,
)


class TestScoring(unittest.TestCase):
    def setUp(self):
        print(f"\n running {type(self).__name__}")

        self.up = pd.Series(list(range(5))).rename("up")
        self.down = pd.Series(list(reversed(range(5)))).rename("down")
        self.up_down = pd.Series([1, 0, 1, 0, 1]).rename("up_down")
        self.down_up = (1 - self.up_down).rename("down_up")
        self.up_float = (self.up / self.up.max()).rename("up_float")
        self.pos_neg = pd.Series([0, -0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]).rename(
            "pos_neg"
        )

    def test_correlation(self):
        assert np.isclose(correlation(self.up, self.up), 1)
        assert np.isclose(correlation(self.up, self.down), -1)
        assert np.isclose(correlation(self.up, self.up_down), 0)
        assert np.isclose(correlation(self.up, self.down_up), 0)

    def test_tie_broken_rank_correlation(self):
        assert np.isclose(tie_broken_rank_correlation(self.up, self.up), 1)
        assert np.isclose(tie_broken_rank_correlation(self.up, self.down), -1)
        # tie_broken_rank_correlation ranks the submission not the targets
        assert np.isclose(tie_broken_rank_correlation(self.up, self.up_down), 0.5)
        assert np.isclose(tie_broken_rank_correlation(self.up, self.down_up), 0.5)
        assert np.isclose(tie_broken_rank_correlation(self.up_down, self.up), 0)
        assert np.isclose(tie_broken_rank_correlation(self.down_up, self.up), 0)

    def test_spearman_correlation(self):
        assert np.isclose(spearman_correlation(self.up, self.up), 1)
        assert np.isclose(spearman_correlation(self.up, self.down), -1)
        assert np.isclose(spearman_correlation(self.up, self.up_down), 0)
        assert np.isclose(spearman_correlation(self.up, self.down_up), 0)
        assert np.isclose(spearman_correlation(self.up_down, self.up), 0)
        assert np.isclose(spearman_correlation(self.down_up, self.up), 0)

    def test_pearson_correlation(self):
        assert np.isclose(pearson_correlation(self.up, self.up), 1)
        assert np.isclose(pearson_correlation(self.up, self.down), -1)
        assert np.isclose(pearson_correlation(self.up, self.up_down), 0)
        assert np.isclose(pearson_correlation(self.up, self.down_up), 0)
        assert np.isclose(pearson_correlation(self.up_down, self.up), 0)
        assert np.isclose(pearson_correlation(self.down_up, self.up), 0)

    def test_tie_broken_rank(self):
        assert np.isclose(
            tie_broken_rank(self.up.to_frame()).T, [0.1, 0.3, 0.5, 0.7, 0.9]
        ).all()
        assert np.isclose(
            tie_broken_rank(self.up_down.to_frame()).T, [0.5, 0.1, 0.7, 0.3, 0.9]
        ).all()

    def test_tie_kept_rank(self):
        assert np.isclose(
            tie_kept_rank(self.up.to_frame()).T, [0.1, 0.3, 0.5, 0.7, 0.9]
        ).all()
        assert np.isclose(
            tie_kept_rank(self.up_down.to_frame()).T, [0.7, 0.2, 0.7, 0.2, 0.7]
        ).all()

    def test_gaussian(self):
        assert np.isclose(
            gaussian(self.up_float).values.T,
            [-np.inf, -0.6744897501960817, 0, 0.6744897501960817, np.inf],
        ).all()

    def test_variance_normalize(self):
        assert np.isclose(
            variance_normalize(self.up_float).values.T,
            [
                0.0,
                0.7071067811865475,
                1.414213562373095,
                2.1213203435596424,
                2.82842712474619,
            ],
        ).all()

    def test_one_hot_encode(self):
        assert np.isclose(
            one_hot_encode(self.up.to_frame(), ["up"]).values.T,
            [
                [1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ],
        ).all()

    def test_power(self):
        assert np.isclose(
            power(self.pos_neg.to_frame(), 1.5),
            [
                [0.0],
                [0.0],
                [0.3535533905932738],
                [-0.3535533905932738],
                [1.0000000000000000],
                [-1.0000000000000000],
                [2.8284271247461903],
                [-2.8284271247461903],
            ],
        ).all()

    def test_tie_kept_rank__gaussianize__pow_1_5(self):
        assert np.isclose(
            tie_kept_rank__gaussianize__pow_1_5(self.up_float.to_frame()),
            [
                [-1.4507885796854221],
                [-0.3797472709071263],
                [0.0000000000000000],
                [0.3797472709071261],
                [1.4507885796854221],
            ],
        ).all()

    def test_orthoganalize(self):
        assert np.isclose(
            orthogonalize(self.up.to_frame().values, self.up.to_frame().values),
            [0, 0, 0, 0, 0],
        ).all()
        assert np.isclose(
            orthogonalize(self.up.to_frame().values, self.up_down.to_frame().values),
            [[-2], [1], [0], [3], [2]],
        ).all()
        assert np.isclose(
            orthogonalize(
                self.down_up.to_frame().values, self.up_down.to_frame().values
            ),
            [[0], [1], [0], [1], [0]],
        ).all()

    def test_stake_weight(self):
        assert np.isclose(
            stake_weight(self.up.to_frame(), pd.Series([1], index=[self.up.name])),
            self.up.values.T,
        ).all()
        assert np.isclose(
            stake_weight(
                pd.concat([self.up, self.down], axis=1),
                pd.Series([1, 1], index=[self.up.name, self.down.name]),
            ),
            ((self.up + self.down) / 2).values.T,
        ).all()

    def test_neutralize(self):
        assert np.isclose(
            neutralize(self.up.to_frame(), pd.DataFrame([0, 0, 0, 0, 0])).values.T,
            self.up - self.up.mean(),
        ).all()
        assert np.isclose(
            neutralize(
                pd.concat([self.up, self.down], axis=1),
                pd.concat(
                    [pd.Series([0, 0, 0, 0, 0]), pd.Series([0, 0, 0, 0, 0])], axis=1
                ),
            ).values.T,
            pd.concat(
                [self.up - self.up.mean(), self.down - self.down.mean()], axis=1
            ).values.T,
        ).all()
        assert np.isclose(
            neutralize(self.up_down.to_frame(), self.down_up.to_frame()).values.T,
            [0, 0, 0, 0, 0],
        ).all()
        # ensure it works for multiple submissions/neutralizers
        assert np.isclose(
            neutralize(
                pd.concat([self.up_down, self.up_down], axis=1),
                pd.concat([self.down_up, self.down_up], axis=1),
            ).values.T,
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ).all()

    def test_numerai_corr_doesnt_clobber_targets(self):
        s = [x/4 for x in range(5)]
        df = pd.DataFrame({
            "target": s,
            "prediction": reversed(s)
        })
        numerai_corr(df[["prediction"]], df["target"])
        assert pd.Series(s).equals(df["target"]), f"{s} != {list(df['target'].values)}"
