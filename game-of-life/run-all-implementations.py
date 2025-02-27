import os
from pathlib import Path

TEST_CASES = [
    # <grid size> <iterations>
    "2048 10000",
    "4096 1000",
    "8192 1000",
    "16384 100",
    "32768 100",
]

def BIT_PACKING_MODE(cu_file):
    id = ID(cu_file)
    no_bit_packing = ['reference/baseline']
    tiled_bit_packing = ['reference/tiled-bitwise-macro']

    if id in no_bit_packing:
        return 0
    if id in tiled_bit_packing:
        return 2
    return 1

def ID(cu_file):
    SCRIPT_DIR = Path(__file__).parent
    CU_FILE_DIR = Path(cu_file).parent
    return str(CU_FILE_DIR.relative_to(SCRIPT_DIR))

BASH_COMMAND = "sbatch compile-and-run.sbatch.sh"

class GolRunner:
    @staticmethod
    def run_all():
        for cu_file in GolRunner._get_all_cu_files():
            GolRunner.run(cu_file)

    @staticmethod
    def run(cu_file):
        print(f"Launching jobs for '{cu_file}'")
        for test_case in TEST_CASES:
            command = f"{BASH_COMMAND} {cu_file} {ID(cu_file)} {BIT_PACKING_MODE(cu_file)} {test_case}"            

            print(f" -- {command}\n -- ", end="")
            os.system(command)

    @staticmethod
    def _get_all_cu_files():
        for path in Path(__file__).parent.rglob("*.cu"):
            yield path

if __name__ == "__main__":
    GolRunner.run_all()