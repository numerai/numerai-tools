import inspect
import unittest

import numpy as np
import pandas as pd  # type: ignore

from numerai_tools.scoring import (
    contribution_scores,
    correlation,
    correlation_contribution,
    filter_sort_neutralizers,
    neutral_corr,
    neutral_meta_model_contribution,
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
    filter_sort_index,
    filter_sort_index_many,
    filter_sort_top_bottom,
    filter_sort_top_bottom_concat,
    alpha,
    meta_portfolio_contribution,
)


def neutral_fixture():
    """Deterministic predictions / neutralizers / meta model / targets used by the
    neutral_corr and neutral_meta_model_contribution tests. The predictions span
    the interesting cases: one column fully explained by a neutralizer, one half
    explained, and one independent of them."""
    rng = np.random.default_rng(0)
    n = 100
    index = [f"id{i:03d}" for i in range(n)]
    neutralizers = pd.DataFrame(
        rng.normal(size=(n, 3)), index=index, columns=["f0", "f1", "f2"]
    )
    predictions = pd.DataFrame(
        {
            "exposed": neutralizers["f0"].values,
            "mixed": 0.5 * neutralizers["f1"].values + 0.5 * rng.normal(size=n),
            "clean": rng.normal(size=n),
        },
        index=index,
    )
    meta_model = pd.Series(rng.normal(size=n), index=index, name="mm")
    # the Jupiter target is already a 5-bucket series in [-2, 2]
    targets = pd.Series(
        rng.choice([-2.0, -1.0, 0.0, 1.0, 2.0], size=n, p=[0.05, 0.2, 0.5, 0.2, 0.05]),
        index=index,
        name="target",
    )
    return predictions, neutralizers, meta_model, targets


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

    def test_filter_sort_index(self):
        # Test with 2 simple ranges with different indices
        s = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
        t = pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])
        new_s, new_t = filter_sort_index(s, t)
        self.assertEqual(len(new_s), 4)
        self.assertEqual(len(new_t), 4)
        self.assertTrue(np.array_equal(new_s.index, [1, 2, 3, 4]))
        self.assertTrue(np.array_equal(new_t.index, [1, 2, 3, 4]))
        self.assertTrue(np.array_equal(new_s.values, [2, 3, 4, 5]))
        self.assertTrue(np.array_equal(new_t.values, [1, 2, 3, 4]))

    def test_filter_sort_index_invalid(self):
        # Ensure assertion error when max filtered ratio is exceeded
        s = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
        t = pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])
        with self.assertRaises(AssertionError):
            filter_sort_index(s, t, max_filtered_ratio=0.1)

    def test_filter_sort_index_many(self):
        # Test with a DataFrame
        s = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
        t = pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])
        new_s, new_t = filter_sort_index_many([s, t])
        self.assertEqual(len(new_s), 4)
        self.assertEqual(len(new_t), 4)
        self.assertTrue(np.array_equal(new_s.index, [1, 2, 3, 4]))
        self.assertTrue(np.array_equal(new_t.index, [1, 2, 3, 4]))
        self.assertTrue(np.array_equal(new_s.values, [2, 3, 4, 5]))
        self.assertTrue(np.array_equal(new_t.values, [1, 2, 3, 4]))

    def test_filter_sort_index_many_invalid(self):
        # Ensure assertion error when max filtered ratio is exceeded
        s = pd.Series([1, 2, 3, 4, 5], index=[0, 1, 2, 3, 4])
        t = pd.Series([1, 2, 3, 4, 5], index=[1, 2, 3, 4, 5])
        with self.assertRaises(AssertionError):
            filter_sort_index_many([s, t], max_filtered_ratio=0.1)

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

    def test_neutralize_basic(self):
        assert np.isclose(
            neutralize(self.up.to_frame(), pd.DataFrame([0, 0, 0, 0, 0])).values.T,
            self.up - self.up.mean(),
        ).all()

    def test_neutralize_multiple_subs(self):
        assert np.isclose(
            neutralize(self.up_down.to_frame(), self.down_up.to_frame()).values.T,
            [0, 0, 0, 0, 0],
        ).all()

    def test_neutralize_multiple_subs_multiple_neutralizers(self):
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

    def test_neutralize_proportion(self):
        # Test with proportion less than 1
        assert np.isclose(
            neutralize(
                self.up.to_frame(), pd.DataFrame([0, 0, 0, 0, 0]), proportion=0.5
            ).values.T,
            (self.up - self.up.mean() * 0.5),
        ).all()

        # Test with proportion equal to 0
        assert np.isclose(
            neutralize(
                self.up.to_frame(), pd.DataFrame([0, 0, 0, 0, 0]), proportion=0
            ).values.T,
            self.up,
        ).all()

    def test_neutralize_with_nans(self):
        # Test with NaNs in input data
        up_with_nans = self.up.copy()
        up_with_nans[2] = np.nan
        self.assertRaisesRegex(
            AssertionError,
            "Data contains NaNs",
            neutralize,
            up_with_nans.to_frame(),
            pd.DataFrame([0, 0, 0, 0, 0]),
        )

    def test_neutralize_large_data(self):
        # Test with larger dataset
        large_data = pd.DataFrame(np.random.randn(1000, 10))
        neutralizers = pd.DataFrame(np.random.randn(1000, 5))
        neutralized = neutralize(large_data, neutralizers)
        assert neutralized.shape == large_data.shape
        assert not np.isnan(neutralized).any().any()

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
        assert abs(corr_w_pow.iloc[0]) > abs(corr_wo_pow.iloc[0])

    def test_filter_top_bottom(self):
        self.assertRaises(
            TypeError,
            filter_sort_top_bottom,
            self.up,
            top_bottom=None,
        )
        np.testing.assert_allclose(
            filter_sort_top_bottom_concat(self.up, top_bottom=2),
            [0, 1, 3, 4],
        )
        top, bot = filter_sort_top_bottom(
            self.up,
            top_bottom=2,
        )
        np.testing.assert_allclose(top, [3, 4])
        np.testing.assert_allclose(bot, [0, 1])

    def test_neutral_corr(self):
        predictions, neutralizers, _, targets = neutral_fixture()
        np.testing.assert_allclose(
            neutral_corr(predictions, neutralizers, targets),
            [0.06491404084580031, 0.08523062609491586, -0.16120414125319232],
        )
        np.testing.assert_allclose(
            neutral_corr(predictions, neutralizers, targets, top_bottom=20),
            [0.08444670918905811, 0.12688143909398297, -0.09916069852822741],
        )

    def test_neutral_corr_is_neutralized_rank_gauss_correlation(self):
        # neutral_corr must be exactly the correlation of the centered targets
        # with the neutralized, rank-gaussianized predictions...
        predictions, neutralizers, _, targets = neutral_fixture()
        neutralized = neutralize(gaussian(tie_kept_rank(predictions)), neutralizers)
        expected = neutralized.apply(
            lambda sub: pearson_correlation(targets - targets.mean(), sub)
        )
        np.testing.assert_allclose(
            neutral_corr(predictions, neutralizers, targets), expected
        )
        # ...and those predictions must have no exposure left to any neutralizer
        for col in neutralized.columns:
            for factor in neutralizers.columns:
                assert abs(neutralized[col].corr(neutralizers[factor])) < 1e-10

    def test_neutral_corr_constant_neutralizer(self):
        # a constant (zero-variance) neutralizer only removes the mean, and
        # pearson correlation is invariant to that, so neutral_corr reduces to a
        # plain rank-gaussianized correlation. That is the only difference
        # between neutral_corr and a neutralization-free score.
        predictions, _, _, targets = neutral_fixture()
        constant = pd.DataFrame(
            {"constant": np.ones(len(predictions))}, index=predictions.index
        )
        expected = gaussian(tie_kept_rank(predictions)).apply(
            lambda sub: pearson_correlation(targets - targets.mean(), sub)
        )
        np.testing.assert_allclose(
            neutral_corr(predictions, constant, targets), expected
        )

    def test_neutral_corr_has_no_pow_1_5(self):
        # numerai_corr powers both the predictions and the targets; neutral_corr
        # powers neither and exposes no target_pow15 flag to turn one back on.
        assert "target_pow15" not in inspect.signature(neutral_corr).parameters
        predictions, _, _, targets = neutral_fixture()
        constant = pd.DataFrame(
            {"constant": np.ones(len(predictions))}, index=predictions.index
        )
        assert not np.allclose(
            neutral_corr(predictions, constant, targets),
            numerai_corr(predictions, targets),
        )

    def test_neutral_corr_is_scale_invariant(self):
        # pearson correlation is scale-invariant, so variance normalizing the
        # neutralized predictions would be a no-op here.
        predictions, neutralizers, _, targets = neutral_fixture()
        np.testing.assert_allclose(
            neutral_corr(predictions, neutralizers, targets),
            neutral_corr(predictions * 100, neutralizers, targets),
        )

    def test_neutral_corr_with_nans(self):
        predictions, neutralizers, _, targets = neutral_fixture()
        # a few missing neutralizer rows are dropped along with their predictions
        holey = neutralizers.copy()
        holey.iloc[:5, 0] = np.nan
        np.testing.assert_allclose(
            neutral_corr(predictions, holey, targets),
            neutral_corr(predictions.iloc[5:], neutralizers.iloc[5:], targets),
        )
        # too many missing rows must raise rather than silently mis-score
        holey.iloc[:50, 0] = np.nan
        self.assertRaises(
            AssertionError, neutral_corr, predictions, holey, targets
        )

    def test_neutral_meta_model_contribution(self):
        predictions, neutralizers, meta_model, targets = neutral_fixture()
        np.testing.assert_allclose(
            neutral_meta_model_contribution(
                predictions, meta_model, neutralizers, targets
            ),
            [0.008166920268846335, 0.056287525943196595, -0.1465734703333085],
        )
        np.testing.assert_allclose(
            neutral_meta_model_contribution(
                predictions, meta_model, neutralizers, targets, top_bottom=20
            ),
            [0.014212728439240246, 0.07892827246673215, -0.13904192918920918],
        )

    def test_neutral_meta_model_contribution_does_not_neutralize_meta_model(self):
        # RESOLVED (T-803): the meta model passed in is the v3NUSWMM, which is
        # already neutral, so only the submissions are neutralized. This test
        # fails if the meta model is neutralized inside the function.
        predictions, neutralizers, meta_model, targets = neutral_fixture()
        scores = neutral_meta_model_contribution(
            predictions, meta_model, neutralizers, targets
        )
        neutral_preds = neutralize(
            gaussian(tie_kept_rank(predictions)), neutralizers
        ).values
        raw_mm = gaussian(tie_kept_rank(meta_model.to_frame()))[meta_model.name]
        neutralized_mm = neutralize(raw_mm.to_frame(), neutralizers)[meta_model.name]
        np.testing.assert_allclose(
            scores,
            contribution_scores(
                orthogonalize(neutral_preds, raw_mm.values),
                targets.copy(),
                predictions,
            ),
        )
        assert not np.allclose(
            scores,
            contribution_scores(
                orthogonalize(neutral_preds, neutralized_mm.values),
                targets.copy(),
                predictions,
            ),
            atol=1e-6,
        )

    def test_neutral_meta_model_contribution_no_variance_normalize(self):
        # variance normalizing the neutralized predictions divides each score by
        # that prediction's own residual std, which restores full-scale
        # contribution to predictions that were mostly neutralizer exposure.
        predictions, neutralizers, meta_model, targets = neutral_fixture()
        neutral_preds = neutralize(gaussian(tie_kept_rank(predictions)), neutralizers)
        raw_mm = gaussian(tie_kept_rank(meta_model.to_frame()))[meta_model.name]
        assert not np.allclose(
            neutral_meta_model_contribution(
                predictions, meta_model, neutralizers, targets
            ),
            contribution_scores(
                orthogonalize(
                    variance_normalize(neutral_preds).values, raw_mm.values
                ),
                targets.copy(),
                predictions,
            ),
            atol=1e-6,
        )

    def test_neutral_meta_model_contribution_with_nans(self):
        predictions, neutralizers, meta_model, targets = neutral_fixture()
        holey = neutralizers.copy()
        holey.iloc[:50, 0] = np.nan
        self.assertRaises(
            AssertionError,
            neutral_meta_model_contribution,
            predictions,
            meta_model,
            holey,
            targets,
        )

    def test_correlation_contribution(self):
        # characterization guard: correlation_contribution and
        # neutral_meta_model_contribution share contribution_scores, so this
        # pins the behavior the shared code must preserve.
        predictions, _, meta_model, targets = neutral_fixture()
        np.testing.assert_allclose(
            correlation_contribution(predictions, meta_model, targets),
            [-0.07921759154900317, 0.15600418598982832, -0.15555182394041875],
        )
        np.testing.assert_allclose(
            correlation_contribution(predictions, meta_model, targets, 20),
            [-0.08827829422628865, 0.33532314469592894, -0.14008793632499197],
        )
        # the [0, 1] target branch still scales those targets into buckets
        np.testing.assert_allclose(
            correlation_contribution(predictions, meta_model, (targets + 2) / 4),
            correlation_contribution(predictions, meta_model, targets + 2),
        )

    def test_filter_sort_neutralizers(self):
        predictions, neutralizers, _, _ = neutral_fixture()
        # the neutralizer universe is allowed to be much larger than the
        # scored universe without tripping the filtered-ratio check
        wide = pd.concat(
            [
                neutralizers,
                neutralizers.rename(index=lambda i: f"extra{i}"),
            ]
        )
        filtered_preds, filtered_neutralizers = filter_sort_neutralizers(
            predictions, wide
        )
        assert filtered_preds.index.equals(predictions.index.sort_values())
        assert filtered_neutralizers.index.equals(predictions.index.sort_values())
        # but dropping too many of the scored ids must raise
        self.assertRaises(
            AssertionError,
            filter_sort_neutralizers,
            predictions,
            neutralizers.iloc[:50],
        )

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
        assert np.isclose(score[0], -0.001580068753957352)
        assert np.isclose(score[1], 0.00237010313093603)


if __name__ == "__main__":
    unittest.main()
