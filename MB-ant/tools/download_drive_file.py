"""Fetch a public Drive research file when gdown encounters a warning page.

Usage: python tools/download_drive_file.py FILE_ID OUTPUT
The fetched file is saved as data and never executed.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
from bs4 import BeautifulSoup


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file_id")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    suffix = args.output.suffix.lower()
    if suffix not in {".mat", ".m", ".py", ".txt"}:
        parser.error("unsupported file extension")
    session = requests.Session()
    url = "https://drive.usercontent.google.com/download"
    response = session.get(url, params={"id": args.file_id, "export": "download"}, timeout=60)
    response.raise_for_status()
    if "text/html" in response.headers.get("content-type", ""):
        form = BeautifulSoup(response.text, "html.parser").find("form")
        if form is None or form.get("action") != url:
            raise RuntimeError("unexpected Google Drive download page")
        params = {item["name"]: item.get("value", "") for item in form.find_all("input")
                  if item.get("name")}
        if params.get("id") != args.file_id or params.get("confirm") != "t":
            raise RuntimeError("unexpected Google Drive confirmation form")
        response = session.get(url, params=params, timeout=60)
        response.raise_for_status()
    if suffix == ".mat":
        if not (response.content.startswith(b"MATLAB ") or
                response.content.startswith(b"\x89HDF\r\n\x1a\n")):
            raise RuntimeError("response is not a MATLAB MAT-file")
    elif "text/html" in response.headers.get("content-type", "") or b"\x00" in response.content:
        raise RuntimeError("response is not a text source file")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(response.content)
    print(f"Saved {args.output} ({len(response.content)} bytes)")


if __name__ == "__main__":
    main()
