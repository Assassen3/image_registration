import os.path
import subprocess

import registration
from msi_proc_for_all import msi_proc
from matrix import T2T
import time

# 定义程序的路径
cpp_program = r"E:\FANG Leisen\xpy_3d_reconstruction\cmake-build-debug\3d.exe"


# 运行C++程序
def run_cpp_program(program_path,path,  csv_path):
    time_T = time.time()
    try:
        result = subprocess.run([program_path, path, csv_path], check=True, text=True, capture_output=True)
        print("C++ 程序输出:", result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"运行 C++ 程序时出错: {e}")
        print(f"错误信息: {e.stderr}")
    print(f"用时: {time.time() - time_T}")

def main(path: str):
    msi_proc(path)
    registration.image_registration(path)
    T2T(path)
    run_cpp_program(cpp_program, path, os.path.join(path, "T.csv"))



if __name__ == "__main__":
    path = r'F:\共享盘\2025.5.26\TEXT\多光谱三维重构\B01'
    main(path)
