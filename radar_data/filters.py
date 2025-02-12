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
