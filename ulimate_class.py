import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class DataFilterError(KeyError):
    pass


class ultimate_class:
    def __init__(self, i: int) -> None:
        self._radar_df = pd.read_csv(f"data/processed data/radar_data_{i}.csv")
        self._lidar_df = pd.read_csv(f"data/processed data/lidar_data_{i}.csv")
        self._radar_df_filtered = self._radar_df
        self._radar_df_deleted = pd.DataFrame({"X, (m)": (), "Y, (m)": ()})

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
            return self._radar_df
        else:
            self._radar_df_filtered = self._radar_df[
                self._filter(filter_id)
            ].reset_index(drop=drop)
            self._radar_df_deleted = self._radar_df[
                ~self._filter(filter_id)
            ].reset_index(drop=drop)
            return self._radar_df_filtered

    def draw(
        self,
        s: float = 0.5,
        figsize: tuple[int, int] = (16, 10),
        lidar_draw=False,
        show=True,
        raw=False,
        clean=False,
    ) -> None:
        plt.figure(figsize=figsize)
        # plt.title(self.name)
        plt.xlabel("OX, (m)")
        plt.ylabel("OY, (m)")
        if raw:
            plt.scatter(
                self._radar_df["X, (m)"],
                self._radar_df["Y, (m)"],
                c=(0, 1, 0, 1),
                s=s,
            )
        else:
            plt.scatter(
                self._radar_df_filtered["X, (m)"],
                self._radar_df_filtered["Y, (m)"],
                c=(0, 1, 0, 1),
                s=s,
            )
            if not clean:
                plt.scatter(
                    self._radar_df_deleted["X, (m)"],
                    self._radar_df_deleted["Y, (m)"],
                    c=(1, 0, 0, 1),
                    s=s,
                )
        if lidar_draw:
            plt.scatter(
                self._lidar_df["X, (m)"],
                self._lidar_df["Y, (m)"],
                c=self._lidar_df["color"],
                s=s,
            )
        if show:
            plt.show()

    def save(self, path: str) -> None:
        plt.savefig(path)
        plt.close()


frame = ultimate_class(95)
for i in range(7):
    frame.filtred(i)
    frame.draw(show=False)
    frame.save(f"data/filtered/filter_{i}_applied(dirty).png")
    frame.draw(show=False, clean=True)
    frame.save(f"data/filtered/filter_{i}_applied(clean).png")
frame.draw(show=False, raw=True)
frame.save(f"data/filtered/raw.png")
