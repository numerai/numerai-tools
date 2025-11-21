import unittest

import numpy as np
import pandas as pd  # type: ignore

from numerai_tools.indexing import (
    filter_sort_index,
    filter_sort_index_many,
    filter_sort_top_bottom,
    filter_sort_top_bottom_concat,
)


class TestIndexing(unittest.TestCase):
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
