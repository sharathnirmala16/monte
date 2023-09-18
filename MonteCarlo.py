import numpy as np
import pandas as pd
import MonteCarloCython as MC


class MonteCarlo:
    __simulations: int
    __forecasts: int

    def __init__(self, simulations: int, forecasts: int) -> None:
        self.__simulations = simulations
        self.__forecasts = forecasts

    def simulate(self, data: pd.Series) -> np.ndarray:
        return MC.simulate(data.values, self.__forecasts, self.__simulations)

    @staticmethod
    def expected_value(paths: np.ndarray) -> float:
        return paths[-1].mean()

    @staticmethod
    def expected_returns(paths: np.ndarray) -> float:
        return ((MonteCarlo.expected_value(paths) / paths[0, 1]) - 1) * 100
