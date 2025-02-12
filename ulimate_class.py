import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from tqdm import tqdm
import json
from collections import Counter


class DataFilterError(KeyError):
    pass


class ultimate_class:
    def __init__(self, i: int) -> None:
        self._radar_df = pd.read_csv(f"data/processed data/radar_data_{i}.csv")
        self._lidar_df = pd.read_csv(f"data/processed data/lidar_data_{i}.csv")
        self._radar_df_filtered = self._radar_df
        self._radar_df_deleted = pd.DataFrame({"X, (m)": (), "Y, (m)": ()})
        self.scene_number = i
        self.filter = -1

    def _filter(self, filter_id: int) -> pd.Series:
        match filter_id:
            # ultimate (all)
            case 0:
                return (
                    (self._radar_df["HasQuality"] == 1.0)
                    & (self._radar_df["QAmbigState"] != 1)
                    & (self._radar_df["QDistLatRMS"] <= 2.16)
                    & (self._radar_df["QDistLongRMS"] <= 4.6)
                    & (self._radar_df["DistanceAccuracy"] <= 0.2)
                )
            # QAmbigState filter
            case 1:
                return (self._radar_df["HasQuality"] == 1.0) & (
                    self._radar_df["QAmbigState"] != 1
                )
            # QDistLatRMS filter
            case 2:
                return (self._radar_df["HasQuality"] == 1.0) & (
                    self._radar_df["QDistLatRMS"] <= 2.16
                )
            # QDistLongRMS filter
            case 3:
                return (self._radar_df["HasQuality"] == 1.0) & (
                    self._radar_df["QDistLongRMS"] <= 4.6
                )
            # DistanceAccuracy filter
            case 4:
                return self._radar_df["DistanceAccuracy"] <= 0.2
            # QPDH0 filter
            case 5:
                return (self._radar_df["HasQuality"] == 1.0) & (
                    self._radar_df["QPDH0"] == 0.25
                )
            # ultimate+QPDH0 filter
            case 6:
                return (
                    (self._radar_df["HasQuality"] == 1.0)
                    & (self._radar_df["QAmbigState"] != 1)
                    & (self._radar_df["QDistLatRMS"] <= 2.16)
                    & (self._radar_df["QDistLongRMS"] <= 4.6)
                    & (self._radar_df["DistanceAccuracy"] <= 0.2)
                    & (self._radar_df["QPDH0"] == 0.25)
                )
            # not a filter_id
            case _:
                raise DataFilterError

    def filtred(self, filter_id: int, change=False, drop=True) -> pd.DataFrame:
        if change:
            self._radar_df = self._radar_df[self._filter(filter_id)].reset_index(
                drop=drop
            )
            self.filter = filter_id
            return self._radar_df
        else:
            self._radar_df_filtered = self._radar_df[
                self._filter(filter_id)
            ].reset_index(drop=drop)
            self._radar_df_deleted = self._radar_df[
                ~self._filter(filter_id)
            ].reset_index(drop=drop)
            self.filter = filter_id
            return self._radar_df_filtered

    def clusterisation(
        self,
        use_filter: bool = True,
        s: float = 3,
        lidar_s: float = 0.5,
        figsize: tuple[int, int] = (16, 10),
        title: str = "Clusteraised RadarData",
        lidar_draw=False,
        show: bool = False,
        path: str = None,
    ) -> pd.DataFrame:
        path = (
            path
            or f"data/clusteraised/scene_{self.scene_number}({self.filter} filter applied).png"
        )
        data = (
            self._radar_df_filtered.copy(False)
            if use_filter
            else self._radar_df.copy(False)
        )

        with open(
            f"data/raw data/radar_positions.json", "r"
        ) as file:  # считываем корды радара
            radar_positions = {float(k): v for k, v in json.load(file).items()}
        v_data = []
        for k, v in radar_positions.items():
            v_data.extend(
                data[data["radar_idx"] == k]["AbsoluteRadialVelocity"]
                / (data[data["radar_idx"] == k]["X, (m)"] - v[0])
                * (
                    (data[data["radar_idx"] == k]["X, (m)"] - v[0]) ** 2
                    + (data[data["radar_idx"] == k]["Y, (m)"] - v[1]) ** 2
                )
                ** 0.5
            )
        data["VAbs, (m/s)"] = v_data

        dots = np.array(
            tuple(
                zip(
                    data["X, (m)"] * (np.abs(data["VAbs, (m/s)"]) > 1.5),
                    data["Y, (m)"] * (np.abs(data["VAbs, (m/s)"]) > 1.5),
                    data["VAbs, (m/s)"] * 4,
                )
            )
        )

        db = DBSCAN(eps=3, min_samples=8).fit(dots)
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel("OX, (m)")
        plt.ylabel("OY, (m)")

        unique_labels = set(db.labels_)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        colors = [
            plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))
        ]
        road = max(Counter(db.labels_).items(), key=lambda x: x[1])[0]

        for k, col in zip(unique_labels, colors):
            if k == -1:
                col = [0, 0, 0, 1]
            elif k == road:
                col = (col[0], col[1], col[2], 0.1)
            class_member_mask = db.labels_ == k
            plt.scatter(
                data[class_member_mask]["X, (m)"],
                data[class_member_mask]["Y, (m)"],
                color=col,
                s=s,
            )

        if lidar_draw:
            plt.scatter(
                self._lidar_df["X, (m)"],
                self._lidar_df["Y, (m)"],
                c=self._lidar_df["color"],
                s=lidar_s,
            )

        if show:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()

        data["ClusterNumber"] = db.labels_
        return data

    def draw(
        self,
        s: float = 0.5,
        s_lidar: float = 0.5,
        figsize: tuple[int, int] = (16, 10),
        title: str = "Filtered RadarData",
        lidar_draw=False,
        show=False,
        raw=False,
        clean=False,
        path: str = None,
    ) -> None:
        path = (
            path
            or f"data/filtered/scene_{self.scene_number}({self.filter} filter applied).png"
        )
        plt.figure(figsize=figsize)
        plt.title(title)
        plt.xlabel("OX, (m)")
        plt.ylabel("OY, (m)")
        if raw:
            plt.scatter(
                self._radar_df["X, (m)"],
                self._radar_df["Y, (m)"],
                color=(0, 1, 0, 1),
                s=s,
            )
        else:
            plt.scatter(
                self._radar_df_filtered["X, (m)"],
                self._radar_df_filtered["Y, (m)"],
                color=(0, 1, 0, 1),
                s=s,
            )
            if not clean:
                plt.scatter(
                    self._radar_df_deleted["X, (m)"],
                    self._radar_df_deleted["Y, (m)"],
                    color=(1, 0, 0, 1),
                    s=s,
                )
        if lidar_draw:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient
            plt.scatter(
                self._lidar_df["X, (m)"],
                self._lidar_df["Y, (m)"],
                c=self._lidar_df["color"],
                s=s_lidar,
            )

        if show:
            plt.show()
        else:
            plt.savefig(path)
            plt.close()


# for i in tqdm(range(100)):
#     frame = ultimate_class(i)
#     frame.filtred(0)
#     frame.draw()
