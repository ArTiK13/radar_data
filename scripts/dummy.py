import shutil
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path
from typing import Optional
from radar_data.parser import parsing_frames
import radar_data.filters
import json

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input-scene", "-i", type=Path, required=True)  # json
    parser.add_argument("--output-scene", "-o", type=Path, required=True)
    parser.add_argument("--filter", "-f", type=str, required=True)
    parser.add_argument("--filter-args", "-a", type=str, required=False)

    args = parser.parse_args()

    # Read scene

    radar_df, lidar_df = parsing_frames(args.input_scene)

    # print(lidar_df, radar_df)

    # Prepare filter with `args.filter` and `args.filter_args`
    scene_filter = getattr(import_module("radar_data.filters"), args.filter)()

    # Apply filter
    radar_df = scene_filter.apply(radar_df)

    # Save scene (just copy for this example)
    radar_tags = """X_RAW, (m)
Y_RAW, (m)
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
    radar_df = radar_df[radar_tags]
    # print(type(lidar_df))
    of = {
        "lidar": tuple(map(tuple, lidar_df.values)),
        "radar": tuple(map(tuple, radar_df.values)),
    }
    # print(of)
    with open(str(args.output_scene), "w") as file:
        json.dump(of, file)
    # radar_df.to_csv(str(args.output_scene).replace(".json", ".csv"))

    # shutil.copy(args.input_scene, args.output_scene)
