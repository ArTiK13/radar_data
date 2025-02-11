import shutil
from argparse import ArgumentParser
from importlib import import_module
from pathlib import Path
from typing import Optional


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--input-scene", "-i", type=Path, required=True)
    parser.add_argument("--output-scene", "-o", type=Path, required=True)
    parser.add_argument("--filter", "-f", type=str, required=True)
    parser.add_argument("--filter-args", "-a", type=str, required=False)

    args = parser.parse_args()

    # Read scene
    ...

    # Prepare filter with `args.filter` and `args.filter_args`
    scene_filter = getattr(import_module("radar_data.filters"), args.filter)()

    # Apply filter
    ...

    # Save scene (just copy for this example)

    shutil.copy(args.input_scene, args.output_scene)
