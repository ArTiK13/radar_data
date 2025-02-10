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
        self, s: float = 0.5, figsize: tuple[int, int] = (16, 10), lidar_draw=False
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
        plt.show()


class Velocity_cololoring(colored_frame_all):
    def color(self, lidar_coloring=False) -> None:
        self._radar_df["color"] = [
            (1, 0, 0.5, 1) for _ in range(len(self._radar_df))
        ]  # красим радар
        if lidar_coloring:
            pass  # красим лидар


v7 = Velocity_cololoring(7, "velocity")
v7.color()
v7.draw()
