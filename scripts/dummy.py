import shutil
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path
from typing import Optional
from scripts.parser import parsing_frame
import scripts.filters
import json

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input-scene", "-i", type=Path, required=True)  # json
    parser.add_argument("--output-scene", "-o", type=Path, required=True)
    parser.add_argument("--filter", "-f", type=str, required=True)
    parser.add_argument("--filter-args", "-a", type=str, required=False)

    args = parser.parse_args()

    radar_df, lidar_df = parsing_frame(args.input_scene)

    scene_filter = getattr(import_module("scripts.filters"), args.filter)()
    if args.filter_args:
        match args.filter:
            case "VelocityFilter":
                radar_df = scene_filter.apply(
                    radar_df, min_velocity=float(args.filter_args)
                )
            case "DeltaTimeFix":
                radar_df = scene_filter.apply(radar_df, deltaT=float(args.filter_args))
            case _:
                radar_df = scene_filter.apply(radar_df)
    else:
        radar_df = scene_filter.apply(radar_df)

    of = {
        "lidar": tuple(map(tuple, lidar_df.values)),
        "radar": tuple(map(tuple, radar_df.values)),
    }
    with open(str(args.output_scene), "w") as file:
        json.dump(of, file)
