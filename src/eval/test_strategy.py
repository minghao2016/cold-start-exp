__author__ = 'shuochang'

from strategy import EntropyStrategy, EntropyZeroStrategy
import pytest
import pandas as pd


class TestEntropy:

    def test_entropy(self):
        count1 = pd.Series(data=[30, 30, 30])
        assert abs(EntropyStrategy.entropy(count1) - 1.09861) < 0.0001
        count2 = pd.Series(data=[30, 10])
        assert abs(EntropyStrategy.entropy(count2) - 0.56233) < 0.0001

    def test_entropy_zero(self):
        count1 = pd.Series(data=[30, 30, 30])
        assert abs(EntropyZeroStrategy.entropy_zero(count1, 5, 100) - 0.21794) < 0.0001
        count1 = pd.Series(data=[50])
        assert abs(EntropyZeroStrategy.entropy_zero(count1, 5, 100) - 0.094520) < 0.0001