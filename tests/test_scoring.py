import unittest

import numpy as np
import pandas as pd

from numerai_tools.scoring import (
    correlation,
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
)


class TestScoring(unittest.TestCase):
    def setUp(self):
        print(f'\n running {type(self).__name__}')

        self.up = pd.Series(list(range(5))).rename('up')
        self.down = pd.Series(list(reversed(range(5)))).rename('down')
        self.up_down = pd.Series([1, 0, 1, 0, 1]).rename('up_down')
        self.down_up = (1 - self.up_down).rename('down_up')
        self.up_float = (self.up / self.up.max()).rename('up_float')
        self.pos_neg = pd.Series([0, -0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]).rename(
            'pos_neg'
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

    def test_neutralize(self):
        reciprocal_std_dev = 1 / self.up_down.values.std()
        assert np.isclose(
            neutralize(self.up_down.to_frame(), self.down_up.to_frame()).values.T,
            [reciprocal_std_dev, 0, reciprocal_std_dev, 0, reciprocal_std_dev],
        ).all()
        # ensure it works for multiple submissions/neutralizers
        assert np.isclose(
            neutralize(
                pd.concat([self.up_down, self.up_down], axis=1),
                pd.concat([self.down_up, self.down_up], axis=1),
            ).values.T,
            [
                [reciprocal_std_dev, 0, reciprocal_std_dev, 0, reciprocal_std_dev],
                [reciprocal_std_dev, 0, reciprocal_std_dev, 0, reciprocal_std_dev],
            ],
        ).all()

    def test_one_hot_encode(self):
        assert np.isclose(
            one_hot_encode(self.up.to_frame(), ['up']).values.T,
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
