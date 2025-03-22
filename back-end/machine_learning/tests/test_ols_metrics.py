import unittest
import pandas as pd
import numpy as np
from machine_learning.ols_metrics import OLSMetrics

class TestCalculateOLS(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(33)
        self.dataset = pd.Series(rng.integers(0, 100, 100))

    def test_calculate_R2(self):

        self.assertEqual(OLSMetrics(self.dataset).calculate_r2(), 0.00209958)

    def test_calculate_coef(self):
        self.assertEqual(OLSMetrics(self.dataset).calculate_coef(), -43876400)
