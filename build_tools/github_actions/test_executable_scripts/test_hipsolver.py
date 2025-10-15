import logging
import os
import shlex
import subprocess
from pathlib import Path

THEROCK_BIN_DIR = os.getenv("THEROCK_BIN_DIR")
SCRIPT_DIR = Path(__file__).resolve().parent
THEROCK_DIR = SCRIPT_DIR.parent.parent.parent

PLATFORM = os.getenv("PLATFORM")
AMDGPU_FAMILIES = os.getenv("AMDGPU_FAMILIES")

# GTest sharding
SHARD_INDEX = os.getenv("SHARD_INDEX", 1)
TOTAL_SHARDS = os.getenv("TOTAL_SHARDS", 1)
envion_vars = os.environ.copy()
# For display purposes in the GitHub Action UI, the shard array is 1th indexed. However for shard indexes, we convert it to 0th index.
envion_vars["GTEST_SHARD_INDEX"] = str(int(SHARD_INDEX) - 1)
envion_vars["GTEST_TOTAL_SHARDS"] = str(TOTAL_SHARDS)

logging.basicConfig(level=logging.INFO)

tests_to_exclude = [
    "*known_bug*",
    "*HEEVD*float_complex*",
    "*HEEVJ*float_complex*",
    "*HEGVD*float_complex*",
    "*HEGVJ*float_complex*",
    "*HEEVDX*float_complex*",
    "*SYTRF*float_complex*",
    "*HEEVD*double_complex*",
    "*HEEVJ*double_complex*",
    "*HEGVD*double_complex*",
    "*HEGVJ*double_complex*",
    "*HEEVDX*double_complex*",
    "*SYTRF*double_complex*",
]

exclusion_list = ":".join(tests_to_exclude)

cmd = [
    f"{THEROCK_BIN_DIR}/hipsolver-test",
    f"--gtest_filter=-{exclusion_list}",
]

logging.info(f"++ Exec [{THEROCK_DIR}]$ {shlex.join(cmd)}")
subprocess.run(cmd, cwd=THEROCK_DIR, check=True, env=envion_vars)
