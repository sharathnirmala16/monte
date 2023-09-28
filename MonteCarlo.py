from typing import Callable

import numpy as np
import pandas as pd
import MonteCarloCython as MC


class MonteCarlo:
    @staticmethod
    def simulate(
        data: np.ndarray,
        forecasts: int,
        simulations: int,
        distribution: Callable = np.random.standard_t,
    ) -> np.ndarray:
        return MC.simulate(data, forecasts, simulations)

    @staticmethod
    def expected_value(data: pd.Series, forecasts: int, simulations: int) -> float:
        paths = MonteCarlo.simulate(data, forecasts, simulations)
        return paths[-1].mean()

    @staticmethod
    def expected_returns(data: pd.Series, forecasts: int, simulations: int) -> float:
        paths = MonteCarlo.simulate(data, forecasts, simulations)
        return ((paths[-1].mean() / paths[0, 1]) - 1) * 100
