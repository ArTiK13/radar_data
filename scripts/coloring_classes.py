import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class colored_frame_radar:
    def __init__(self, i: int, color_name: str) -> None:
        self._radar_df = pd.read_csv(f"data/processed data/radar_data_{i}.csv")
        self.name = color_name

    def draw(self, s: float = 0.5, figsize: tuple[int, int] = (16, 10)) -> None:
        plt.figure(figsize=figsize)
        plt.title(self.name)
        plt.xlabel("OX, (m)")
        plt.ylabel("OY, (m)")
        plt.scatter(
            self._radar_df["X, (m)"],
            self._radar_df["Y, (m)"],
            c=self._radar_df["color"],
            s=s,
        )
        plt.show()


class colored_frame_lidar:
    def __init__(self, i: int, color_name: str) -> None:
        self._lidar_df = pd.read_csv(f"data/processed data/lidar_data_{i}.csv")
        self.name = color_name

    def draw(self, s: float = 0.5, figsize: tuple[int, int] = (16, 10)):
        plt.figure(figsize=figsize)
        plt.title(self.name)
        plt.xlabel("OX, (m)")
        plt.ylabel("OY, (m)")
        plt.scatter(
            self._lidar_df["X, (m)"],
            self._lidar_df["Y, (m)"],
            c=self._lidar_df["color"],
            s=s,
        )


class colored_frame_all:
    def __init__(self, i: int, color_name: str) -> None:
        self._radar_df = pd.read_csv(f"data/processed data/radar_data_{i}.csv")
        self._lidar_df = pd.read_csv(f"data/processed data/lidar_data_{i}.csv")
        self.name = color_name

    def draw(
        self, s: float = 0.5, figsize: tuple[int, int] = (16, 10), lidar_draw=False, show = True
    ) -> None:
        plt.figure(figsize=figsize)
        plt.title(self.name)
        plt.xlabel("OX, (m)")
        plt.ylabel("OY, (m)")
        plt.scatter(
            self._radar_df["X, (m)"],
            self._radar_df["Y, (m)"],
            c=self._radar_df["color"],
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


class Velocity_cololoring(colored_frame_all):
    def color(self, lidar_coloring=False) -> None:
        self._radar_df["color"] = [
            (1, 0, 0.5, 1) for _ in range(len(self._radar_df))
        ]  # красим радар
        if lidar_coloring:
            pass  # красим лидар


class radar_idx_cololoring(colored_frame_all):
    def _filtering(self) -> pd.Series:
        return self._radar_df["QPDH0"] > 0

    def filtred(self) -> pd.DataFrame:
        self._radar_df_filtered = self._radar_df[self._filtering]
        return self._radar_df_filtered

    def color(self, lidar_coloring=False) -> None:
        radar_color = {
            1: (1, 0, 0, 1),
            2: (1, 0, 1, 1),
            3: (0, 1, 0, 1),
            4: (0, 1, 1, 1),
            7: (0, 0, 1, 1),
        }
        self._radar_df["color"] = [
            radar_color[self._radar_df["radar_idx"][i]]
            for i in range(len(self._radar_df))
        ]  # красим радар
        if lidar_coloring:
            pass  # красим лидар


class QAmbigState_cololoring(colored_frame_all):
    def _filtering(self) -> pd.Series:
        return self._radar_df["HasQuality"] == 1.0

    def filtred(self, change=True) -> pd.DataFrame:
        if change:
            self._radar_df = self._radar_df[self._filtering()].reset_index(drop=True)
            return self._radar_df
        else:
            self._radar_df_filtered = self._radar_df[self._filtering()]
            return self._radar_df_filtered

    def color(self, lidar_coloring=False) -> None:
        self.filtred()
        QAmbigState_color = {
            1.0: (1, 0, 0, 1),
            2.0: (0, 1, 0, 1),
            3.0: (0, 0, 1, 1),
        }
        self._radar_df["color"] = [
            QAmbigState_color[self._radar_df["QAmbigState"][i]]
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            pass  # красим лидар


# v7 = QAmbigState_cololoring(52, "QAmbigState")
# v7.color()
# v7.draw()

for i in range(100):
    v = QAmbigState_cololoring(i, 'QAmbSt')
    v.color()
    v.draw(show=False)
    v.save(f'data/QAmbingState/frame_{i}.png')