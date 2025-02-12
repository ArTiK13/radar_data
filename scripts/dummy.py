import shutil
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path
from typing import Optional
from radar_data.parser import parsing_frames
import radar_data.filters


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input-scene", "-i", type=Path, required=True)  # json
    parser.add_argument("--output-scene", "-o", type=Path, required=True)
    parser.add_argument("--filter", "-f", type=str, required=True)
    parser.add_argument("--filter-args", "-a", type=str, required=False)

    args = parser.parse_args()

    # Read scene

    radar_df, lidar_df = parsing_frames(args.input_scene)

    print(lidar_df, radar_df)

    # Prepare filter with `args.filter` and `args.filter_args`
    scene_filter = getattr(import_module("radar_data.filters"), args.filter)()

    # Apply filter
    radar_df = scene_filter.apply(radar_df)

    # Save scene (just copy for this example)
    radar_df.to_csv(str(args.output_scene).replace(".json", ".csv"))

    # shutil.copy(args.input_scene, args.output_scene)
