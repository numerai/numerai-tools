import unittest

import numpy as np
import pandas as pd  # type: ignore

from numerai_tools.data import (
    one_hot_encode,
    balanced_rank_transform,
    quantile_bin,
)


class TestData(unittest.TestCase):
    def setUp(self):
        self.up = pd.Series(list(range(5))).rename("up")
        self.down = pd.Series(list(reversed(range(5)))).rename("down")
        self.up_down = pd.Series([1, 0, 1, 0, 1]).rename("up_down")
        self.down_up = (1 - self.up_down).rename("down_up")
        self.up_float = (self.up / self.up.max()).rename("up_float")
        self.pos_neg = pd.Series([0, -0, 0.5, -0.5, 1.0, -1.0, 2.0, -2.0]).rename(
            "pos_neg"
        )

    def test_balanced_rank_transform_basic_single_group(self):
        df = pd.DataFrame({"x": [10, 20, 30]})
        result = balanced_rank_transform(df, cols=["x"], rank_group=None)
        expected = pd.Series([(1 - 0.5) / 3, (2 - 0.5) / 3, (3 - 0.5) / 3], name="x")
        pd.testing.assert_series_equal(result["x"], expected)

    def test_balanced_rank_transform_grouping_is_respected(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "x": [1, 100, 2, 200],
            }
        )
        result = balanced_rank_transform(df, cols=["x"], rank_group="group")
        # ranks independently within groups
        group_A_vals = result.loc[df["group"] == "A", "x"].tolist()
        group_B_vals = result.loc[df["group"] == "B", "x"].tolist()

        assert group_A_vals == [(1 - 0.5) / 2, (2 - 0.5) / 2]
        assert group_B_vals == [(1 - 0.5) / 2, (2 - 0.5) / 2]

    def test_balanced_rank_transform_filter(self):
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "x": [1, 100, 2, 200],
                "include": [True, False, True, False],
            }
        )
        result = balanced_rank_transform(
            df, cols=["x"], rank_group="group", rank_filter="include"
        )
        # only included rows ranked
        expected = pd.Series([0.5, 0.5], index=[0, 2], name="x")
        pd.testing.assert_series_equal(result["x"], expected)

    def test_balanced_rank_transform_nan_handling(self):
        df = pd.DataFrame({"group": ["A", "A", "B", "B"], "x": [1, None, 2, 200]})
        result = balanced_rank_transform(df, cols=["x"], rank_group="group")

        expected = pd.Series([0.5, np.nan, (1 - 0.5) / 2, (2 - 0.5) / 2], name="x")
        pd.testing.assert_series_equal(result["x"], expected)

    def test_balanced_rank_transform_index_alignment(self):
        df = pd.DataFrame({"group": ["A", "B"], "x": [10, 20]})
        result = balanced_rank_transform(df, cols=["x"], rank_group="group")
        assert all(result.index == df.index)

    def test_quantile_bin_single_column_series(self):
        s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        binned = quantile_bin(s)
        assert isinstance(binned, pd.DataFrame)
        assert binned.shape[0] == s.shape[0]
        # values should be within 0, 0.25, 0.5, 0.75, 1
        assert set(binned.iloc[:, 0].unique()).issubset({0, 0.25, 0.5, 0.75, 1})

    def test_quantile_bin_multi_column_dataframe(self):
        df = pd.DataFrame(
            {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]}
        )
        binned = quantile_bin(df)
        assert binned.shape == df.shape
        for col in binned.columns:
            assert set(binned[col].unique()).issubset({0, 0.25, 0.5, 0.75, 1})

    def test_quantile_bin_nan_handling(self):
        df = pd.DataFrame(
            {"a": [1, np.nan, 3, 4, np.nan], "b": [np.nan, 2, 3, np.nan, 5]}
        )
        binned = quantile_bin(df)
        assert np.isnan(binned.loc[1, "a"])
        assert np.isnan(binned.loc[0, "b"])

    def test_quantile_bin_all_identical_values(self):
        df = pd.DataFrame({"a": [5, 5, 5, 5, 5]})
        binned = quantile_bin(df)
        assert (binned["a"] == 0).all()

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


if __name__ == "__main__":
    unittest.main()
