"""Run the complete optimizer and network test suites.

Usage:
    python test.py

The script runs regular unit/CUDA tests and then launches the two-process CPU
DDP tests through torchrun.
"""

from pathlib import Path
import os
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
ENV = {**os.environ, "PYTHONPATH": str(ROOT)}


def run(command):
    """Run one test command and stop immediately if it fails."""
    print(f"\n$ {' '.join(map(str, command))}", flush=True)
    subprocess.run(command, cwd=ROOT, env=ENV, check=True)


def main():
    run([
        sys.executable,
        "-m",
        "compileall",
        "-q",
        "runtime",
        "optimizer",
        "network",
        "model",
        "callbacks",
        "tests",
    ])
    run([
        sys.executable,
        "-m",
        "pytest",
        "-q",
        "tests/test_optimizer.py",
        "tests/test_network.py",
        "tests/test_model.py",
        "tests/test_runtime.py",
        "tests/test_gpu_ddp.py",
    ])

    torchrun = shutil.which("torchrun")
    if torchrun is None:
        torchrun_command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
        ]
    else:
        torchrun_command = [torchrun]

    for script in ("tests/test_ddp.py", "tests/test_network_ddp.py"):
        run(torchrun_command + ["--standalone", "--nproc_per_node=2", script])
    print("\nAll active package and DDP tests passed.")


if __name__ == "__main__":
    main()
