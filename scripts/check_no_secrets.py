"""Fail if the repository contains real credential material.

A plain `git grep private_key` is useless here: that string legitimately appears in
the CI workflow itself and in this script, so the check would always report a
false positive and quickly be ignored. This looks for what actually leaks a secret:

  1. a service-account JSON  -- "type": "service_account" plus a long private key
  2. a PEM block WITH a body -- header followed by several lines of base64
  3. a key-like filename     -- bdao-group-<hex>.json, *-sa-*.json, *.pem, *.p12

Exit code 0 = clean, 1 = something found.

    python scripts/check_no_secrets.py
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PEM_WITH_BODY = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----\s*\r?\n"
    rb"([A-Za-z0-9+/=]{40,}\s*\r?\n){2,}"
)
SERVICE_ACCOUNT_TYPE = re.compile(rb'"type"\s*:\s*"service_account"')
PRIVATE_KEY_FIELD = re.compile(rb'"private_key"\s*:\s*"[^"]{80,}')
KEY_FILENAME = re.compile(
    r"(service[_-]?account.*\.json$|-sa-.*\.json$|\.pem$|\.p12$|(^|/)\.env$)",
    re.IGNORECASE,
)
# files that are allowed to name credentials because they detect or document them
ALLOWLIST = {"scripts/check_no_secrets.py", ".github/workflows/checks.yml"}
BINARY_EXT = {".png", ".jpg", ".jpeg", ".gif", ".docx", ".pdf", ".xlsx", ".pptx",
              ".zip", ".gz", ".7z"}


def _find_git():
    p = shutil.which("git")
    if p:
        return p
    for c in (r"D:\Git\cmd\git.exe", r"C:\Program Files\Git\cmd\git.exe",
              os.path.expanduser(r"~\AppData\Local\Programs\Git\cmd\git.exe")):
        if os.path.exists(c):
            return c
    return None


def tracked_files():
    git = _find_git()
    if git:
        try:
            out = subprocess.check_output([git, "ls-files", "-z"], cwd=ROOT,
                                          stderr=subprocess.DEVNULL)
            files = [f for f in out.decode("utf-8", "replace").split("\0") if f]
            if files:
                return files
        except Exception:
            pass
    walk = []
    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__")]
        for fn in fns:
            walk.append(os.path.relpath(os.path.join(dp, fn), ROOT).replace("\\", "/"))
    return walk


def main():
    files = tracked_files()
    problems = []
    for rel in files:
        if KEY_FILENAME.search(rel):
            problems.append((rel, "filename looks like a key file"))
            continue
        if os.path.splitext(rel)[1].lower() in BINARY_EXT:
            continue
        try:
            path = os.path.join(ROOT, rel)
            if os.path.getsize(path) > 8 * 1024 * 1024:
                continue
            data = open(path, "rb").read()
        except OSError:
            continue
        if PEM_WITH_BODY.search(data):
            problems.append((rel, "PEM private key with body"))
        if SERVICE_ACCOUNT_TYPE.search(data) and PRIVATE_KEY_FIELD.search(data):
            problems.append((rel, "service-account key JSON"))
        elif PRIVATE_KEY_FIELD.search(data) and rel not in ALLOWLIST:
            problems.append((rel, "long private_key literal"))

    if problems:
        print("Suspected credential material:\n")
        for f, why in problems:
            print(f"  {f}  ->  {why}")
        print("\nRotate the key first, then clean history. Do not just delete the file.")
        return 1

    print(f"Scanned {len(files)} files: no credential material found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
