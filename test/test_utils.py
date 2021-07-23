import unittest
import numpy as np

from utils.functions import eval_metrics


class TestMethods(unittest.TestCase):

    def test_eval_metrics(self):
        rmse, mae, _ = eval_metrics(actual=[3, -0.5, 2, 7], pred=[2.5, 0.0, 2, 8])
        self.assertEqual(rmse, np.sqrt(0.375))
        self.assertEqual(mae, 0.5)

        rmse, mae, r2 = eval_metrics(actual=[0.0, 0.0, 0.0, 0.0], pred=[0, 0.0, 0.0, 0.0])
        self.assertEqual(rmse, 0.0)
        self.assertEqual(mae, 0.0)
        self.assertEqual(rmse, 0.0)

        _, _, r2 = eval_metrics(actual=[1.0, 2.0, 3.0],
                                pred=[1.0, 2.0, 3.0])
        self.assertEqual(r2, 1.0)


if __name__ == '__main__':
    unittest.main()