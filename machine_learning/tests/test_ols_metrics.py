import unittest
import pandas as pd
import numpy as np
from machine_learning.ols_metrics import calculate_coef, calculate_r2

class TestCalculateOLS(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(33)
        self.dataset = pd.Series(rng.integers(0, 100, 100))

    def test_calculate_R2(self):
        self.assertEqual(calculate_r2(self.dataset), 0.0021)

    def test_calculate_coef(self):
        self.assertEqual(calculate_coef(self.dataset), -0.0438764)