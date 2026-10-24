"""Fill the street cache by re-running the (resumable) fetchers a few times.

Overpass intermittently 504s, so a single pass rarely completes every metro. Each
pass only fetches what is still missing, so repeated passes make monotonic
progress. Two window sizes are filled:

  08d_fetch_block_streets.py   small 0.45 km window  -> the street-level maps
  08c_fetch_streets.py         wide  2.6 km window   -> the cluster/district maps

Afterwards rerun analysis/08b_metro_maps.py to pick up the streets.
"""

from __future__ import annotations

import os
import subprocess
import sys

PY = r"C:\Users\ASUS\.dsh\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe"
HERE = os.path.dirname(os.path.abspath(__file__))

SCRIPTS = ["08d_fetch_block_streets.py", "08c_fetch_streets.py"]
for script in SCRIPTS:
    print(f"########## {script}", flush=True)
    for attempt in range(1, 7):
        print(f"=== pass {attempt} ===", flush=True)
        r = subprocess.run([PY, "-u", os.path.join(HERE, script)],
                           capture_output=True, text=True, encoding="utf-8",
                           errors="replace")
        out = r.stdout or ""
        for line in out.strip().splitlines()[-4:]:
            print("   ", line, flush=True)
        if "incomplete" not in out and "budget hit" not in out and "not writing" not in out:
            print("complete", flush=True)
            break
print("done", flush=True)
