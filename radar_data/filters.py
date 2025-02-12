from abc import ABC, abstractmethod
from typing import Sequence

from .types import Scene


class BaseFilter(ABC):
    def __init__(self):
        # TODO: decice how to configure filters
        pass

    @abstractmethod
    def apply(self, scene: Scene) -> Scene:
        pass


class Compose(BaseFilter):
    """
    Compose several fitlers

    Given (filter_1, filter_2, ..., fitler_n) applies them sequentially
    """

    def __init__(self, filters: Sequence[BaseFilter]):
        self.filters = filters

    def apply(self, scene: Scene) -> Scene:
        for filter_ in self.filters:
            scene = filter_.apply(scene)
        return scene


class IdentityFilter(BaseFilter):
    apply = lambda self, scene: scene


class QAmbigStateFilter(BaseFilter):
    def apply(self, scene: Scene) -> Scene:
        return scene[(scene["HasQuality"] == 1.0) & (scene["QAmbigState"] != 1)]


class DistanceAccuracyFilter(BaseFilter):
    def apply(self, scene: Scene) -> Scene:
        return scene[scene["DistanceAccuracy"] <= 0.2]


class QDistLatRMSFilter(BaseFilter):
    def apply(self, scene: Scene) -> Scene:
        return scene[(scene["HasQuality"] == 1.0) & (scene["QDistLatRMS"] <= 2.16)]


class QDistLongRMSFilter(BaseFilter):
    def apply(self, scene: Scene) -> Scene:
        return scene[(scene["HasQuality"] == 1.0) & (scene["QDistLongRMS"] <= 4.6)]


class QPDH0Filter(BaseFilter):
    def apply(self, scene: Scene) -> Scene:
        return scene[(scene["HasQuality"] == 1.0) & (scene["QPDH0"] == 0.25)]


class UltimateFilter(BaseFilter):
    def apply(self, scene: Scene) -> Scene:
        return scene[
            (
                (scene["HasQuality"] == 1.0)
                & (scene["QAmbigState"] != 1)
                & (scene["QDistLatRMS"] <= 2.16)
                & (scene["QDistLongRMS"] <= 4.6)
                & (scene["DistanceAccuracy"] <= 0.2)
                & (scene["QPDH0"] == 0.25)
            )
        ]


class VelocityFilter(BaseFilter):
    def apply(self, scene: Scene) -> Scene:
        min_velocity = 2

        radar_positions = {
            "1": [4.856, 1.29, 3.24],
            "2": [4.856, -1.29, 3.24],
            "3": [5.103, 1.23, 3.23],
            "4": [5.103, -1.23, 3.23],
            "7": [5.139, 0.332, 0.635],
        }
        v_data = []
        for k, v in radar_positions.items():
            v_data.extend(
                scene[scene["radar_idx"] == k]["AbsoluteRadialVelocity"]
                / (scene[scene["radar_idx"] == k]["X, (m)"] - v[0])
                * (
                    (scene[scene["radar_idx"] == k]["X, (m)"] - v[0]) ** 2
                    + (scene[scene["radar_idx"] == k]["Y, (m)"] - v[1]) ** 2
                )
                ** 0.5
            )
        v_data = np.array(v_data)
        return scene[(np.abs(v_data) > min_velocity)]
