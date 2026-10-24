"""Fail if any relative link or embedded image in the markdown does not resolve.

This repository's whole value is the documents, and they cross-reference figures
and each other constantly. A renamed chart or a translated document silently breaks
a link, and on GitHub that shows up as a dead link nobody notices. This turns that
into a CI failure.

Checks, for every relative target in `[text](target)` and `![alt](target)`:
  * path targets must exist on disk
  * `#anchor` targets are ignored (GitHub generates them, we do not re-implement it)
  * `http(s):` and `mailto:` are skipped

Exit code 0 = all links resolve, 1 = at least one is broken.

    python scripts/check_links.py
"""

from __future__ import annotations

import os
import re
import sys

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
SKIP_PREFIX = ("http://", "https://", "mailto:", "#", "tel:")


def main():
    broken = []
    checked = 0
    docs = 0

    for dp, dns, fns in os.walk(ROOT):
        dns[:] = [d for d in dns if d not in (".git", "__pycache__")]
        for fn in fns:
            if not fn.endswith(".md"):
                continue
            docs += 1
            path = os.path.join(dp, fn)
            rel_doc = os.path.relpath(path, ROOT).replace("\\", "/")
            text = open(path, encoding="utf-8").read()
            for m in LINK.finditer(text):
                target = m.group(1).strip()
                if not target or target.startswith(SKIP_PREFIX):
                    continue
                # strip an optional title: [x](path "title")
                target = target.split()[0]
                target = target.split("#")[0]
                if not target:
                    continue
                checked += 1
                resolved = os.path.normpath(os.path.join(dp, target))
                if not os.path.exists(resolved):
                    broken.append((rel_doc, m.group(1).strip()))

    print(f"Checked {checked} relative links across {docs} markdown files.")
    if broken:
        print(f"\n{len(broken)} broken link(s):\n")
        for doc, target in broken:
            print(f"  {doc}  ->  {target}")
        return 1
    print("All relative links resolve.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
