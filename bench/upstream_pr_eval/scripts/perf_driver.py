"""Runs perf_run.py configs in fresh subprocesses with a 180s cap each.
Usage: perf_driver.py <venv_python> <section> <impl1,impl2> <T1,T2,...>"""
import subprocess
import sys
from pathlib import Path

py, section = sys.argv[1], sys.argv[2]
impls = sys.argv[3].split(",")
Ts = [int(t) for t in sys.argv[4].split(",")]
base = str(Path(__file__).resolve().parent)

for T in Ts:
    for impl in impls:
        cmd = [py, f"{base}/perf_run.py", section, impl, str(T)]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=185)
            out = r.stdout.strip()
            if r.returncode != 0:
                print(f"RESULT section={section} impl={impl} T={T} ERROR:\n{r.stderr.strip()[-1500:]}")
            else:
                print(out.splitlines()[-1] if out else f"(no output) {r.stderr[-500:]}")
        except subprocess.TimeoutExpired:
            print(f"RESULT section={section} impl={impl} T={T} DNF (>180s)")
        sys.stdout.flush()
