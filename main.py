import subprocess
import time
from pathlib import Path

from registration import T2T, image_registration, split_ms_images

cpp_program = Path(__file__).parent / '3d' / '3d.exe'


def run_cpp_program(program_path: str, base_folder: str, csv_path: str):
    time_start = time.time()
    try:
        result = subprocess.run([program_path, base_folder, csv_path], check=True, text=True, capture_output=True)
        print("3D reconstruction stdout:\n", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"3D reconstruction error:\n{e}")
    print(f"Time: {time.time() - time_start}")


def run(base_path: Path | str):
    base_path = Path(base_path)

    split_ms_images(base_path)
    image_registration(base_path)
    T2T(base_path)
    run_cpp_program(str(cpp_program.absolute()), str(base_path), str(base_path / 'T.csv'))


if __name__ == "__main__":
    path = Path(r"C:\Users\EY\Desktop\K1")
    run(path)
