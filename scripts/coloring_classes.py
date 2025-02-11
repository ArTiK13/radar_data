import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# наверное потом удалить, т.к. один фрейм не тяжелый и можно подругражуть сразу и лидар и радар

# class colored_frame_radar:
#     def __init__(self, i: int, color_name: str) -> None:
#         self._radar_df = pd.read_csv(f"data/processed data/radar_data_{i}.csv")
#         self.name = color_name

#     def draw(self, s: float = 0.5, figsize: tuple[int, int] = (16, 10)) -> None:
#         plt.figure(figsize=figsize)
#         plt.title(self.name)
#         plt.xlabel("OX, (m)")
#         plt.ylabel("OY, (m)")
#         plt.scatter(
#             self._radar_df["X, (m)"],
#             self._radar_df["Y, (m)"],
#             c=self._radar_df["color"],
#             s=s,
#         )
#         plt.show()


# class colored_frame_lidar:
#     def __init__(self, i: int, color_name: str) -> None:
#         self._lidar_df = pd.read_csv(f"data/processed data/lidar_data_{i}.csv")
#         self.name = color_name

#     def draw(self, s: float = 0.5, figsize: tuple[int, int] = (16, 10)):
#         plt.figure(figsize=figsize)
#         plt.title(self.name)
#         plt.xlabel("OX, (m)")
#         plt.ylabel("OY, (m)")
#         plt.scatter(
#             self._lidar_df["X, (m)"],
#             self._lidar_df["Y, (m)"],
#             c=self._lidar_df["color"],
#             s=s,
#         )

# parrent class


class colored_frame_all:
    def __init__(self, i: int, color_name: str) -> None:
        self._radar_df = pd.read_csv(f"data/processed data/radar_data_{i}.csv")
        self._lidar_df = pd.read_csv(f"data/processed data/lidar_data_{i}.csv")
        self.name = color_name

    def filtred(self, change=True, drop=True) -> pd.DataFrame:
        if change:
            self._radar_df = self._radar_df[self._filtering()].reset_index(drop=drop)
            return self._radar_df
        else:
            self._radar_df_filtered = self._radar_df[self._filtering()]
            return self._radar_df_filtered

    def draw(
        self,
        s: float = 0.5,
        figsize: tuple[int, int] = (16, 10),
        lidar_draw=False,
        show=True,
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


# основные классы покрасок


class test_cololoring(colored_frame_all):  # тестовая, показывает как красить
    def color(self, lidar_coloring=False) -> None:
        self._radar_df["color"] = [
            (1, 0, 0.5, 1) for _ in range(len(self._radar_df))
        ]  # красим радар
        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class radar_idx_cololoring(
    colored_frame_all
):  # покраска по радарам, пользы нет, визуально видно раздереление радаров
    def _filtering(self) -> pd.Series:
        return self._radar_df["QPDH0"] > 0

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
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class QAmbigState_cololoring(
    colored_frame_all
):  # ебать вообще какое полезное просто имба
    def _filtering(self) -> pd.Series:
        return (self._radar_df["HasQuality"] == 1.0) & (self._radar_df["QPDH0"] == 0.25)

    def color(self, lidar_coloring=False) -> None:
        self.filtred()
        QAmbigState_color = {
            1.0: (1, 0, 0, 0),
            2.0: (0, 1, 0, 1),
            3.0: (0, 0, 1, 1),
        }
        self._radar_df["color"] = [
            QAmbigState_color[self._radar_df["QAmbigState"][i]]
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class QVLongRMS_cololoring(
    colored_frame_all
):  ## бесполезно, т.к. убирает много хороших точек
    def _filtering(self) -> pd.Series:
        return (
            self._radar_df["HasQuality"] == 1.0
        )  # & (self._radar_df["QPDH0"] == 0.25)

    def color(self, lidar_coloring=False) -> None:
        self.filtred()
        # c_generator = lambda i: (((i-0.25)/2)**0.2, (1-(i-0.25)/2)**0.2, 0, 1)
        c_generator = lambda i: (
            (1, 0, 0, 1) if i < 0.3 else ((0, 1, 0, 1) if i < 0.4 else (0, 0, 1, 1))
        )
        self._radar_df["color"] = [
            c_generator(self._radar_df["QVLongRMS"][i])
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class QDistLatRMS_cololoring(
    colored_frame_all
):  # полезно, красные точки (самая большая погрешность), релаьно выбросы
    def _filtering(self) -> pd.Series:
        return (
            self._radar_df["HasQuality"] == 1.0
        )  # & (self._radar_df["QPDH0"] == 0.25)

    def color(self, lidar_coloring=False) -> None:
        self.filtred()
        # c_generator = lambda i: (((i-0.25)/2)**0.2, (1-(i-0.25)/2)**0.2, 0, 1)
        c_generator = lambda i: (
            (1, 0, 0, 1)
            if i > 2.16
            else (
                0,
                ((i - 0.6159) / 1.0812) ** 0.52,
                (1 - (i - 0.6159) / 1.0812) ** 0.52,
                1,
            )
        )
        self._radar_df["color"] = [
            c_generator(self._radar_df["QDistLatRMS"][i])
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class QDistLongRMS_cololoring(
    colored_frame_all
):  # аналогично красные выбросы, синие кайф, зеленые в целом норм
    def _filtering(self) -> pd.Series:
        return (
            self._radar_df["HasQuality"] == 1.0
        )  # & (self._radar_df["QPDH0"] == 0.25)

    def color(self, lidar_coloring=False) -> None:
        self.filtred()
        # c_generator = lambda i: (((i-0.25)/2)**0.2, (1-(i-0.25)/2)**0.2, 0, 1)
        c_generator = lambda i: (
            (1, 0, 0, 1)
            if i > 4.6
            else (
                0,
                ((i - 0.6159) / 3.015) ** 0.52,
                (1 - (i - 0.6159) / 3.015) ** 0.52,
                1,
            )
        )
        self._radar_df["color"] = [
            c_generator(self._radar_df["QDistLongRMS"][i])
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class Range_cololoring(colored_frame_all):  # что это вообще.....
    def _filtering(self) -> pd.Series:
        return (self._radar_df["QAmbigState"] != 1.0) & (
            self._radar_df["DistanceAccuracy"] < 0.2
        )

    def color(self, lidar_coloring=False) -> None:
        self.filtred()
        Range_cololor = {
            1.0: (1, 0, 0, 1),
            0.0: (0, 1, 0, 1),
        }
        self._radar_df["color"] = [
            Range_cololor[self._radar_df["Range"][i]]
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class DistanceAccuracy_cololoring(colored_frame_all):  # +1 полезность, красные - мусор
    def _filtering(self) -> pd.Series:
        return self._radar_df["QAmbigState"] != 1.0

    def color(self, lidar_coloring=False) -> None:
        self.filtred()
        DistanceAccuracy_cololor = {
            True: (1, 0, 0, 0),
            False: (0, 1, 0, 1),
        }
        self._radar_df["color"] = [
            DistanceAccuracy_cololor[self._radar_df["DistanceAccuracy"][i] > 0.2]
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class AngleAccuracy_cololoring(
    colored_frame_all
):  # забавно показывает погрешности, но выбрасывать по нему нет смысла особо, все норм лежит
    def _filtering(self) -> pd.Series:
        return self._radar_df["QAmbigState"] != 1.0

    def color(self, lidar_coloring=False) -> None:
        # self.filtred()
        colorer = lambda i: (
            (0, 0, 1, 1)
            if i > 0.08
            else (
                (0.01 ** (12.5 * i) - 1) / (-0.99),
                1 - (0.01 ** (12.5 * i) - 1) / (-0.99),
                0,
                1,
            )
        )
        self._radar_df["color"] = [
            colorer(self._radar_df["AngleAccuracy"][i])
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class DynProp_cololoring(colored_frame_all):  # useless
    def _filtering(self) -> pd.Series:
        return (self._radar_df["HasQuality"] == 1.0) & (self._radar_df["QPDH0"] == 0.25)

    def color(self, lidar_coloring=False) -> None:
        # self.filtred()
        colorer = lambda i: (
            (0, 0, 1, 1) if i == 0.0 else ((1, 0, 0, 1) if i == 2.0 else (0, 1, 0, 1))
        )
        self._radar_df["color"] = [
            colorer(self._radar_df["DynProp"][i]) for i in range(len(self._radar_df))
        ]  # красим радар
        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


class RadarCrossSection_cololoring(colored_frame_all):  # ничего не дает
    def _filtering(self) -> pd.Series:
        return self._radar_df["QAmbigState"] != 1.0

    def color(self, lidar_coloring=False) -> None:
        self.filtred()
        colorer = lambda i: (
            1 / (1 + 2.718 ** (0.3 * (10 - i))),
            1 - 1 / (1 + 2.718 ** (0.3 * (10 - i))),
            0,
            1,
        )

        self._radar_df["color"] = [
            colorer(self._radar_df["RadarCrossSection"][i])
            for i in range(len(self._radar_df))
        ]  # красим радар

        if lidar_coloring:
            gradient = []
            for i in range(len(self._lidar_df)):
                color = (self._lidar_df["r, (reflectance)"][i] / 255) ** 0.3
                gradient.append((color, 0, 1 - color, 0.021))
            self._lidar_df["color"] = gradient


# v7 = RadarCrossSection_cololoring(52, "Range")
# v7.color()
# v7.draw()

# for i in range(1):
#     v = QAmbigState_cololoring(i, 'QAmbSt')
#     v.color()
#     v.draw(show=False)
#     v.save(f'data/QAmbingState/frame_{i}.png')


import cmd
import os


class FileManagerCLI(cmd.Cmd):
    prompt = ">> "

    def __init__(self):
        super().__init__()
        self.current_directory = os.getcwd()

    def do_QDistLongRMS(self, line):
        keys = line.split()
        i = int(keys[-2])
        v7 = QDistLongRMS_cololoring(i, "QDistLongRMS")
        v7.color()
        v7.draw()
        v7.save(keys[-1])
        print("Сохранено по адресу", keys[-1])

    def do_QAmbingState(self, line):
        keys = line.split()
        i = int(keys[-2])
        v7 = QAmbingState_cololoring(i, "QAmbingState")
        v7.color()
        v7.draw()
        v7.save(keys[-1])
        print("Сохранено по адресу", keys[-1])

    

    def do_quit(self, line):
        return True

    def postcmd(self, stop, line):
        print()
        return stop


if __name__ == "__main__":
    FileManagerCLI().cmdloop()
