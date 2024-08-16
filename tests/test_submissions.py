import unittest

import numpy as np
import pandas as pd

from numerai_tools.submissions import clean_predictions


class TestSubmissions(unittest.TestCase):
    def setUp(self):
        print(f"\n running {type(self).__name__}")

        self.id_col = "id"
        self.live_ids = pd.Series(list(range(5))).rename(self.id_col)
        self.submission = pd.DataFrame.from_dict(
            {self.id_col: self.live_ids, "prediction": self.live_ids}
        )

    def test_clean_predictions(self):
        assert (
            (
                clean_predictions(
                    self.live_ids,
                    self.submission,
                    id_col="id",
                    rank_and_fill=False,
                ).reset_index()
                == self.submission
            )
            .all()
            .all()
        )

        assert np.isclose(
            clean_predictions(
                self.live_ids,
                self.submission,
                id_col="id",
                rank_and_fill=True,
            ).values.T,
            [[0.1, 0.3, 0.5, 0.7, 0.9]],
        ).all()
