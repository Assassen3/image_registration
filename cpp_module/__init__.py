import os
from pathlib import Path

libs_dir = Path(__file__).parent.resolve()
os.add_dll_directory(str(libs_dir))

from .registration import save_points, load_points
