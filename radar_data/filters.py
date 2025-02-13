from abc import ABC, abstractmethod
from typing import Sequence

from .types import Scene

import numpy as np
from sklearn.cluster import DBSCAN


class BaseFilter(ABC):
    def __init__(self):
        # TODO: decice how to configure filters
        pass

    @abstractmethod
    def apply(self, scene: Scene, **kwargs) -> Scene:
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
                & (np.abs(scene["X, (m)"]) < 150)
            )
        ]


class VelocityFilter(BaseFilter):
    def apply(self, scene: Scene, min_velocity: float = 2.0) -> Scene:
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
                scene[scene["radar_idx"] == int(k)]["AbsoluteRadialVelocity"]
                / (scene[scene["radar_idx"] == int(k)]["X, (m)"] - v[0])
                * (
                    (scene[scene["radar_idx"] == int(k)]["X, (m)"] - v[0]) ** 2
                    + (scene[scene["radar_idx"] == int(k)]["Y, (m)"] - v[1]) ** 2
                )
                ** 0.5
            )
        v_data = np.array(v_data)

        return scene[np.abs(v_data) > min_velocity]


class DeltaTimeFix(BaseFilter):
    def apply(self, scene: Scene, deltaT: float = 0.06) -> Scene:
        radar_positions = {
            "1": [4.856, 1.29, 3.24],
            "2": [4.856, -1.29, 3.24],
            "3": [5.103, 1.23, 3.23],
            "4": [5.103, -1.23, 3.23],
            "7": [5.139, 0.332, 0.635],
        }

        # вычитаем из координат точек координаты радара
        for (
            i,
            cords,
        ) in radar_positions.items():
            for j, ax in enumerate(("X, (m)", "Y, (m)")):
                scene.loc[scene["radar_idx"] == i, ax] -= cords[j]

        # делаем пересчет координат с учетом временного сдвига
        vector_length: pd.Series = (
            scene["X, (m)"] ** 2 + scene["Y, (m)"] ** 2
        ) ** 0.5  # высчитываем точки радара
        rad_del = (deltaT - scene["(radar_point_ts - lidar_ts), (s)"]) * scene[
            "AbsoluteRadialVelocity"
        ]
        scene["X, (m)"] = scene["X, (m)"] * (vector_length + rad_del) / vector_length
        scene["Y, (m)"] = scene["Y, (m)"] * (vector_length + rad_del) / vector_length

        # прибавляем к координатам точек координаты радара
        for (
            i,
            cords,
        ) in radar_positions.items():
            for j, ax in enumerate(("X, (m)", "Y, (m)")):
                scene.loc[scene["radar_idx"] == i, ax] += cords[j]

        return scene


class DeltaTimeFixPredict(BaseFilter):
    def apply(self, scene: Scene, deltaT: float = 0.06) -> Scene:
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
                scene[scene["radar_idx"] == int(k)]["AbsoluteRadialVelocity"]
                / (scene[scene["radar_idx"] == int(k)]["X, (m)"] - v[0])
                * (
                    (scene[scene["radar_idx"] == int(k)]["X, (m)"] - v[0]) ** 2
                    + (scene[scene["radar_idx"] == int(k)]["Y, (m)"] - v[1]) ** 2
                )
                ** 0.5
            )
        v_data = np.array(v_data)

        scene["X, (m)"] = (
            scene["X, (m)"]
            + (deltaT - scene["(radar_point_ts - lidar_ts), (s)"]) * v_data
        )

        return scene[np.abs(scene["X, (m)"]) < 150]


class ClusterisationFilter(BaseFilter):
    def apply(self, scene: Scene, deltaT: float = 0.06) -> Scene:
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
                scene[scene["radar_idx"] == int(k)]["AbsoluteRadialVelocity"]
                / (scene[scene["radar_idx"] == int(k)]["X, (m)"] - v[0])
                * (
                    (scene[scene["radar_idx"] == int(k)]["X, (m)"] - v[0]) ** 2
                    + (scene[scene["radar_idx"] == int(k)]["Y, (m)"] - v[1]) ** 2
                )
                ** 0.5
            )
        v_data = np.array(v_data)

        dots = np.array(
            tuple(
                zip(
                    scene["X, (m)"] * (np.abs(v_data) > 1.5),
                    scene["Y, (m)"] * (np.abs(v_data) > 1.5),
                    v_data * 4,
                )
            )
        )

        db = DBSCAN(eps=3, min_samples=8).fit(dots)

        return scene[db.labels_ != -1]
