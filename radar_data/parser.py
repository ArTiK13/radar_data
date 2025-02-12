import pandas as pd
import json
import struct
from collections import defaultdict
from tqdm import tqdm


def parsing_LEGACY():
    ## 1
    data_primary_raw = pd.read_csv("data/raw data/data.csv")
    data_secondary_raw = pd.read_csv("data/raw data/data_secondary.csv")

    datatypes = {7: 4, 1: 1}
    point_step = 64
    tags_raw = (
        {b"name": "x", b"datatype": 7, b"count": 1, b"offset": 0},
        {b"name": "y", b"datatype": 7, b"count": 1, b"offset": 4},
        {b"name": "z", b"datatype": 7, b"count": 1, b"offset": 8},
        {b"name": "obj_vrel_long", b"datatype": 7, b"count": 1, b"offset": 12},
        {b"name": "obj_lat_speed", b"datatype": 7, b"count": 1, b"offset": 16},
        {b"name": "obj_rcs_value", b"datatype": 7, b"count": 1, b"offset": 20},
        {b"name": "radial_speed_absolute", b"datatype": 7, b"count": 1, b"offset": 24},
        {b"name": "distance_accuracy", b"datatype": 7, b"count": 1, b"offset": 28},
        {b"name": "angle_accuracy", b"datatype": 7, b"count": 1, b"offset": 32},
        {b"name": "pdh0", b"datatype": 7, b"count": 1, b"offset": 36},
        {b"name": "dist_long_rms", b"datatype": 7, b"count": 1, b"offset": 40},
        {b"name": "dist_lat_rms", b"datatype": 7, b"count": 1, b"offset": 44},
        {b"name": "v_long_rms", b"datatype": 7, b"count": 1, b"offset": 48},
        {b"name": "v_lat_rms", b"datatype": 7, b"count": 1, b"offset": 52},
        {b"name": "dyn_prop", b"datatype": 1, b"count": 1, b"offset": 56},
        {b"name": "range", b"datatype": 1, b"count": 1, b"offset": 57},
        {b"name": "has_quality", b"datatype": 1, b"count": 1, b"offset": 58},
        {b"name": "invalid", b"datatype": 1, b"count": 1, b"offset": 60},
        {b"name": "ambig", b"datatype": 1, b"count": 1, b"offset": 59},
    )
    tags = {i[b"name"]: (datatypes[i[b"datatype"]], i[b"offset"]) for i in tags_raw}

    parsed_keys = (
        ("ride_date", "rover", "message_ts", "log_time", "ride_time")
        + tuple("primary_" + tag for tag in tags.keys())
        + tuple("secondary_" + tag for tag in tags.keys())
    )
    preff = ("primary_", "secondary_")
    ## 2
    parsed = defaultdict(list)
    for data_N, data in enumerate(
        (
            data_primary_raw["b'radar_primary_list'"],
            data_secondary_raw["b'radar_secondary_list'"],
        )
    ):
        for i, date_i in enumerate(data):
            dots_raw = bytes(map(int, date_i[1:-1].split(", ")))

            for k in ("ride_date", "rover", "message_ts", "log_time", "ride_time"):
                parsed[k].extend(
                    [(data_primary_raw if not data_N else data_secondary_raw)[str(k.encode())][i]]
                    * (len(dots_raw) // 64)
                )

            for j in range(len(dots_raw) // 64):

                for tag_key, (tag_len, tag_offs) in tags.items():
                    match tag_len:
                        case 4:
                            parsed[tag_key].append(
                                struct.unpack(
                                    "f",
                                    dots_raw[j * 64 + tag_offs : j * 64 + tag_offs + tag_len],
                                )[0]
                            )
                        case 1:
                            parsed[tag_key].append(dots_raw[j * 64 + tag_offs : j * 64 + tag_offs + tag_len].decode())
    ## 3
    parsed_df = pd.DataFrame(parsed)
    parsed_df.to_csv("data/processed data/clean_data_LEGACY.csv", index=False)


def parsing_frames(data_raw_cnt=100, delta_t=0.06):
    lidar_tags = """X, (m)
Y, (m)
Z, (m)
r, (reflectance)
lidar_ring""".split(
        "\n"
    )
    radar_tags = """X, (m)
Y, (m)
Z, (m)
AbsoluteRadialVelocity
RadarCrossSection
RelativeRadialVelocity
RelativeLateralVelocity
Range
DistanceAccuracy
AngleAccuracy
DynProp
HasQuality
QPDH0
QDistLongRMS
QDistLatRMS
QVLongRMS
QVLatRMS
QAmbigState
QInvalidState
(radar_point_ts - lidar_ts), (s)
radar_idx""".split(
        "\n"
    )

    with open(f"data/raw data/radar_positions.json", "r") as file:  # считываем корды радара
        radar_positions = {float(k): v for k, v in json.load(file).items()}

    for jj in tqdm(range(data_raw_cnt)):
        with open(f"data/raw data/scene_{jj}.json", "r") as file:
            radar_lidar_data_raw = json.load(file)
            lidar_df = pd.DataFrame(radar_lidar_data_raw["lidar"], columns=lidar_tags)
            radar_df = pd.DataFrame(radar_lidar_data_raw["radar"], columns=radar_tags)
        radar_df["X_RAW, (m)"], radar_df["Y_RAW, (m)"] = (
            radar_df["X, (m)"],
            radar_df["Y, (m)"],
        )
        for (
            i,
            cords,
        ) in radar_positions.items():  # вычитаем из координат точек координаты радара
            for j, ax in enumerate(("X, (m)", "Y, (m)")):
                radar_df.loc[radar_df["radar_idx"] == i, ax] -= cords[j]

        vector_length: pd.Series = (
            radar_df["X, (m)"] ** 2 + radar_df["Y, (m)"] ** 2
        ) ** 0.5  # высчитываем точки радара
        rad_del = (delta_t - radar_df["(radar_point_ts - lidar_ts), (s)"]) * radar_df["AbsoluteRadialVelocity"]
        radar_df["X, (m)"] = radar_df["X, (m)"] * (vector_length + rad_del) / vector_length
        radar_df["Y, (m)"] = radar_df["Y, (m)"] * (vector_length + rad_del) / vector_length

        for (
            i,
            cords,
        ) in radar_positions.items():  # добавляем к новым координатам корды радаров
            for j, ax in enumerate(("X, (m)", "Y, (m)")):
                radar_df.loc[radar_df["radar_idx"] == i, ax] += cords[j]

        radar_df.to_csv(f"data/processed data/radar_data_{jj}.csv", index=False)
        lidar_df.to_csv(f"data/processed data/lidar_data_{jj}.csv", index=False)


def parsing_frame(path_to_json: str):
    lidar_tags = """X, (m)
Y, (m)
Z, (m)
r, (reflectance)
lidar_ring""".split(
        "\n"
    )
    radar_tags = """X, (m)
Y, (m)
Z, (m)
AbsoluteRadialVelocity
RadarCrossSection
RelativeRadialVelocity
RelativeLateralVelocity
Range
DistanceAccuracy
AngleAccuracy
DynProp
HasQuality
QPDH0
QDistLongRMS
QDistLatRMS
QVLongRMS
QVLatRMS
QAmbigState
QInvalidState
(radar_point_ts - lidar_ts), (s)
radar_idx""".split(
        "\n"
    )

    with open(path_to_json, "r") as file:
        radar_lidar_data_raw = json.load(file)
        lidar_df = pd.DataFrame(radar_lidar_data_raw["lidar"], columns=lidar_tags)
        radar_df = pd.DataFrame(radar_lidar_data_raw["radar"], columns=radar_tags)

    return radar_df, lidar_df


def parsing_frames_aio(data_raw_cnt=100, delta_t=0.06):
    lidar_tags = """X, (m)
Y, (m)
Z, (m)
r, (reflectance)
lidar_ring""".split(
        "\n"
    )
    radar_tags = """X, (m)
Y, (m)
Z, (m)
AbsoluteRadialVelocity
RadarCrossSection
RelativeRadialVelocity
RelativeLateralVelocity
Range
DistanceAccuracy
AngleAccuracy
DynProp
HasQuality
QPDH0
QDistLongRMS
QDistLatRMS
QVLongRMS
QVLatRMS
QAmbigState
QInvalidState
(radar_point_ts - lidar_ts), (s)
radar_idx""".split(
        "\n"
    )

    radar_lidar_data_raw = defaultdict(list)
    for i in tqdm(range(data_raw_cnt)):
        with open(f"data/raw data/scene_{i}.json", "r") as file:
            radar_lidar_data_i = json.load(file)
            radar_lidar_data_raw["lidar"].extend(radar_lidar_data_i["lidar"])
            radar_lidar_data_raw["radar"].extend(radar_lidar_data_i["radar"])

    radar_df = pd.DataFrame(radar_lidar_data_raw["radar"], columns=radar_tags)

    with open(f"data/raw data/radar_positions.json", "r") as file:  # считываем корды радара
        radar_positions = {float(k): v for k, v in json.load(file).items()}

    radar_df["X_RAW, (m)"], radar_df["Y_RAW, (m)"] = (
        radar_df["X, (m)"],
        radar_df["Y, (m)"],
    )
    for (
        i,
        cords,
    ) in radar_positions.items():  # вычитаем из координат точек координаты радара
        for j, ax in enumerate(("X, (m)", "Y, (m)")):
            radar_df.loc[radar_df["radar_idx"] == i, ax] -= cords[j]

    vector_length: pd.Series = (radar_df["X, (m)"] ** 2 + radar_df["Y, (m)"] ** 2) ** 0.5  # высчитываем точки радара
    rad_del = (delta_t - radar_df["(radar_point_ts - lidar_ts), (s)"]) * radar_df["AbsoluteRadialVelocity"]
    radar_df["X, (m)"] = radar_df["X, (m)"] * (vector_length + rad_del) / vector_length
    radar_df["Y, (m)"] = radar_df["Y, (m)"] * (vector_length + rad_del) / vector_length

    for (
        i,
        cords,
    ) in radar_positions.items():  # добавляем к новым координатам корды радаров
        for j, ax in enumerate(("X, (m)", "Y, (m)")):
            radar_df.loc[radar_df["radar_idx"] == i, ax] += cords[j]

    lidar_df = pd.DataFrame(radar_lidar_data_raw["lidar"], columns=lidar_tags)

    radar_df.to_csv("data/processed data/radar_data.csv", index=False)
    lidar_df.to_csv("data/processed data/lidar_data.csv", index=False)


# parsing_LEGACY()
parsing_frames("data/raw data/scene_0.json")
# parsing_frames_aio()
