#!/usr/bin/env python3
"""
Download and extract up to N samples per selected EcoSet category.

EcoSet is distributed as a password-protected zip. Provide the password via:
  - --password "..."
  - or env var ECOSET_PASSWORD

Example:
  python ecoset_sample_download.py \
    --categories banana,chair,extinguisher \
    --split train \
    --n 200 \
    --out_dir ./ecoset_samples \
    --seed 0

If ecoset.zip doesn't exist, the script will download it.
By default, the zip is stored inside out_dir to keep everything together.

Windows speed tip:
  Install aria2c for fastest multi-connection downloads:
    winget install aria2.aria2
  Then run with:
    --downloader auto   (default)
"""

import argparse
import csv
import os
import random
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ECOSET_ZIP_URL = "https://files.ikw.uni-osnabrueck.de/ml/ecoset/ecoset.zip"
IMG_EXTS = (".jpg", ".jpeg", ".JPG", ".JPEG")


def _human_bytes(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    x = float(n)
    for u in units:
        if x < 1024.0:
            return f"{x:.1f}{u}"
        x /= 1024.0
    return f"{x:.1f}PB"


def _head_info(url: str, timeout: int = 60) -> tuple[int | None, bool]:
    req = Request(url, method="HEAD")
    with urlopen(req, timeout=timeout) as resp:
        headers = resp.headers
        cl = headers.get("Content-Length")
        accept_ranges = (headers.get("Accept-Ranges") or "").lower() == "bytes"
        return (int(cl) if cl is not None else None), accept_ranges


def _which(cmd: str) -> str | None:
    return shutil.which(cmd)


def _run(cmd: list[str]) -> None:
    # Use shell=False for safety; capture output only on failure.
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"--- stdout ---\n{p.stdout}\n"
            f"--- stderr ---\n{p.stderr}\n"
        )


def _download_aria2(url: str, dest: Path, connections: int, piece_mb: int, aria2c_path: str | None) -> None:
    aria2 = aria2c_path or _which("aria2c")
    if not aria2:
        raise RuntimeError("aria2c not found on PATH (install aria2 or pass --aria2c_path).")

    aria2p = Path(aria2)
    if not aria2p.exists():
        raise RuntimeError(f"aria2c not found at: {aria2} (check --aria2c_path).")

    dest.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(aria2p),
        "-c",
        "-x", str(max(1, connections)),
        "-s", str(max(1, connections)),
        "-k", f"{max(1, piece_mb)}M",
        "--file-allocation=none",
        "--summary-interval=5",
        "--console-log-level=notice",
        url,
        "-d", str(dest.parent),
        "-o", str(dest.name),
    ]
    print(f"[dl] Using aria2c ({connections} connections): {dest}")
    _run(cmd)


def _download_curl(url: str, dest: Path, max_retries: int) -> None:
    curl = _which("curl")
    if not curl:
        raise RuntimeError("curl not found on PATH")

    dest.parent.mkdir(parents=True, exist_ok=True)

    # curl resume: -C - (continue at last byte)
    # retries handle transient failures
    cmd = [
        curl,
        "-L",
        "-C", "-",
        "--retry", str(max_retries),
        "--retry-delay", "2",
        "--retry-connrefused",
        "-o", str(dest),
        url,
    ]
    print(f"[dl] Using curl (resume+retry): {dest}")
    _run(cmd)


def _download_bits(url: str, dest: Path) -> None:
    # BITS is very reliable on Windows, sometimes faster than Python,
    # but not multi-connection. Still a good fallback.
    dest.parent.mkdir(parents=True, exist_ok=True)

    ps = _which("powershell") or _which("pwsh")
    if not ps:
        raise RuntimeError("PowerShell not found on PATH")

    # Start-BitsTransfer doesn't naturally resume partial files the way curl does;
    # it will restart the transfer if file exists. We keep it as fallback.
    # Use -ErrorAction Stop so failures raise.
    script = (
        f"$ErrorActionPreference='Stop'; "
        f"Import-Module BitsTransfer; "
        f"Start-BitsTransfer -Source '{url}' -Destination '{str(dest)}'"
    )
    cmd = [ps, "-NoProfile", "-Command", script]
    print(f"[dl] Using PowerShell BITS: {dest}")
    _run(cmd)


def _download_python(
    url: str,
    dest: Path,
    chunk_size: int,
    timeout: int,
    max_retries: int,
    retry_backoff: float,
    progress_secs: float,
    verify_size: bool = True,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")

    if dest.exists() and dest.stat().st_size > 0:
        print(f"[ok] Using existing zip: {dest} ({_human_bytes(dest.stat().st_size)})")
        return

    try:
        expected_size, accept_ranges = _head_info(url, timeout=timeout)
    except Exception:
        expected_size, accept_ranges = None, False

    # best-effort disk warning
    try:
        free = shutil.disk_usage(str(dest.parent)).free
        if expected_size is not None and free < expected_size * 1.10:
            print(
                f"[warn] Free space in {dest.parent} is {_human_bytes(free)}, "
                f"EcoSet zip is ~{_human_bytes(expected_size)}. You may run out of disk."
            )
    except Exception:
        pass

    offset = part.stat().st_size if part.exists() else 0
    if offset > 0 and not accept_ranges:
        print("[warn] Server does not advertise Range support; restarting Python download.")
        try:
            part.unlink()
        except Exception:
            pass
        offset = 0

    attempt = 0
    while True:
        attempt += 1
        try:
            headers = {}
            if offset > 0:
                headers["Range"] = f"bytes={offset}-"
            req = Request(url, headers=headers)

            print(
                f"[dl] Using Python (single connection): {dest}\n"
                f"     mode: {'resume' if offset > 0 else 'fresh'}"
                + (f" | expected: {_human_bytes(expected_size)}" if expected_size else "")
            )

            with urlopen(req, timeout=timeout) as resp:
                status = getattr(resp, "status", None)
                if offset > 0 and status == 200:
                    print("[warn] Range ignored; restarting Python download from scratch.")
                    try:
                        part.unlink()
                    except Exception:
                        pass
                    offset = 0
                    continue

                mode = "ab" if offset > 0 else "wb"
                downloaded = offset
                t0 = time.time()
                last = t0

                with open(part, mode) as f:
                    while True:
                        chunk = resp.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)

                        now = time.time()
                        if progress_secs > 0 and (now - last) >= progress_secs:
                            if expected_size:
                                pct = 100.0 * downloaded / expected_size
                                mbps = (downloaded - offset) / max(1e-6, (now - t0)) / (1024 * 1024)
                                print(
                                    f"     ... {_human_bytes(downloaded)} / {_human_bytes(expected_size)} "
                                    f"({pct:.1f}%) | {mbps:.1f} MB/s"
                                )
                            else:
                                mbps = (downloaded - offset) / max(1e-6, (now - t0)) / (1024 * 1024)
                                print(f"     ... {_human_bytes(downloaded)} | {mbps:.1f} MB/s")
                            last = now

            final_size = part.stat().st_size
            if verify_size and expected_size is not None and final_size != expected_size:
                raise RuntimeError(
                    f"Downloaded size mismatch: got {_human_bytes(final_size)} "
                    f"but expected {_human_bytes(expected_size)} (partial kept)."
                )

            part.replace(dest)
            print(f"[ok] Download complete: {dest} ({_human_bytes(dest.stat().st_size)})")
            return

        except (HTTPError, URLError, TimeoutError, RuntimeError) as e:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Python download failed after {attempt} attempts.\n"
                    f"Last error: {e}\n"
                    f"Partial (if any): {part if part.exists() else 'none'}"
                )
            sleep_s = min(60.0, retry_backoff ** (attempt - 1))
            print(f"[warn] Python download error (attempt {attempt}/{max_retries}): {e}")
            print(f"       Retrying in {sleep_s:.1f}s ...")
            time.sleep(sleep_s)
            offset = part.stat().st_size if part.exists() else 0


def download_if_needed(
    url: str,
    zip_path: Path,
    downloader: str,
    connections: int,
    piece_mb: int,
    timeout: int,
    max_retries: int,
    chunk_mb: int,
    progress_secs: float,
    verify_size: bool,
    aria2_path: str
) -> None:
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if zip_path.exists() and zip_path.stat().st_size > 0:
        print(f"[ok] Using existing zip: {zip_path} ({_human_bytes(zip_path.stat().st_size)})")
        return

    # Decide downloader
    order = []
    dl = downloader.lower()

    if dl == "auto":
        # Prefer aria2c (fastest), then curl (resume+retry), then BITS, then Python
        order = ["aria2", "curl", "bits", "python"]
    else:
        order = [dl]

    last_err = None
    for method in order:
        try:
            if method == "aria2":
                _download_aria2(url, zip_path, connections=connections, piece_mb=piece_mb, aria2c_path=aria2_path)
                return
            if method == "curl":
                _download_curl(url, zip_path, max_retries=max_retries)
                return
            if method == "bits":
                _download_bits(url, zip_path)
                return
            if method == "python":
                _download_python(
                    url,
                    zip_path,
                    chunk_size=max(1, chunk_mb) * 1024 * 1024,
                    timeout=timeout,
                    max_retries=max_retries,
                    retry_backoff=1.8,
                    progress_secs=progress_secs,
                    verify_size=verify_size,
                )
                return
            raise ValueError(f"Unknown downloader: {method}")
        except Exception as e:
            last_err = e
            print(f"[warn] Downloader '{method}' failed: {e}")

    raise RuntimeError(f"All download methods failed. Last error: {last_err}")


def normalize_categories(cats_raw: str) -> list[str]:
    cats = []
    for c in cats_raw.split(","):
        c = c.strip()
        if c:
            cats.append(c)
    if not cats:
        raise ValueError("No categories provided.")
    return cats


def folder_matches_label(folder_name: str, label: str) -> bool:
    base = folder_name.strip("/").split("/")[-1]
    if base == label:
        return True
    if base.endswith("_" + label):
        return True
    return base.lower() == label.lower() or base.lower().endswith("_" + label.lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", required=True,
                    help="Comma-separated list of EcoSet labels (e.g., banana,chair,extinguisher)")
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                    help="Which split to sample from")
    ap.add_argument("--n", type=int, default=100,
                    help="Max number of images per category to extract")
    ap.add_argument("--out_dir", default="ecoset_samples",
                    help="Output directory (also used as default download location for ecoset.zip)")
    ap.add_argument("--zip_path", default=None,
                    help="Path to ecoset.zip. If omitted, defaults to <out_dir>/ecoset.zip")
    ap.add_argument("--password", default=None,
                    help="Zip password (or set env ECOSET_PASSWORD)")
    ap.add_argument("--seed", type=int, default=0,
                    help="Random seed for sampling")
    ap.add_argument("--skip_existing", action="store_true",
                    help="If set, don't overwrite existing extracted files")

    # Download controls (Windows-friendly)
    ap.add_argument("--downloader", default="auto", choices=["auto", "aria2", "curl", "bits", "python"],
                    help="Download method. 'auto' prefers aria2->curl->bits->python.")
    ap.add_argument("--connections", type=int, default=16,
                    help="aria2: number of connections/splits (default 16)")
    ap.add_argument("--piece_mb", type=int, default=8,
                    help="aria2: piece size MB (default 8)")
    ap.add_argument("--timeout", type=int, default=60,
                    help="Network timeout seconds (python/head) (default 60)")
    ap.add_argument("--max_retries", type=int, default=8,
                    help="Retries for curl/python (default 8)")
    ap.add_argument("--chunk_mb", type=int, default=64,
                    help="Python: chunk size MB (default 64)")
    ap.add_argument("--progress_secs", type=float, default=5.0,
                    help="Python: progress print interval seconds (default 5; set 0 to disable)")
    ap.add_argument("--no_verify_size", action="store_true",
                    help="Disable size verification (not recommended)")
    ap.add_argument("--aria2c_path", default=None,
                help="Full path to aria2c.exe (overrides PATH lookup).")


    args = ap.parse_args()

    categories = normalize_categories(args.categories)
    split = args.split
    n = max(0, args.n)
    out_dir = Path(args.out_dir)

    # Zip stored in out_dir by default
    if args.zip_path is None:
        zip_path = out_dir / "ecoset.zip"
    else:
        zp = Path(args.zip_path)
        zip_path = (out_dir / zp) if not zp.is_absolute() else zp

    password = args.password or os.environ.get("ECOSET_PASSWORD")

    if n == 0:
        print("[warn] n=0, nothing to extract.")
        return

    if not password:
        print(
            "[error] EcoSet zip is password-protected. Provide it with --password\n"
            "        or set env var ECOSET_PASSWORD."
        )
        sys.exit(2)

    random.seed(args.seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download (fastest possible on Windows if aria2 is installed)
    download_if_needed(
        ECOSET_ZIP_URL,
        zip_path,
        downloader=args.downloader,
        connections=args.connections,
        piece_mb=args.piece_mb,
        timeout=args.timeout,
        max_retries=args.max_retries,
        chunk_mb=args.chunk_mb,
        progress_secs=args.progress_secs,
        verify_size=(not args.no_verify_size),
        aria2_path=args.aria2c_path
    )

    manifest_path = out_dir / f"manifest_{split}_n{n}.csv"
    extracted_rows = []

    print(f"[info] Opening zip: {zip_path}")
    try:
        zf = zipfile.ZipFile(zip_path, "r")
    except zipfile.BadZipFile as e:
        raise RuntimeError(f"Bad zip file: {zip_path}\n{e}")

    split_prefix = f"{split}/"
    members = [m for m in zf.namelist() if m.startswith(split_prefix) and m.lower().endswith(IMG_EXTS)]
    if not members:
        print(f"[error] No image files found under '{split_prefix}' in the zip.")
        sys.exit(3)

    by_folder: dict[str, list[str]] = {}
    for m in members:
        parts = m.split("/")
        if len(parts) < 3:
            continue
        folder = "/".join(parts[:2])
        by_folder.setdefault(folder, []).append(m)

    label_to_folders: dict[str, list[str]] = {lab: [] for lab in categories}
    for folder in by_folder.keys():
        for lab in categories:
            if folder_matches_label(folder, lab):
                label_to_folders[lab].append(folder)

    for lab, folders in label_to_folders.items():
        if not folders:
            print(f"[warn] No folders matched label '{lab}' in split '{split}'.")
        elif len(folders) > 1:
            print(f"[warn] Multiple folders matched label '{lab}': {folders} (sampling across all).")

    for lab, folders in label_to_folders.items():
        lab_members = []
        for f in folders:
            lab_members.extend(by_folder.get(f, []))
        lab_members = sorted(set(lab_members))
        if not lab_members:
            continue

        k = min(n, len(lab_members))
        chosen = random.sample(lab_members, k)

        lab_out = out_dir / split / lab
        lab_out.mkdir(parents=True, exist_ok=True)

        print(f"[x] {lab}: extracting {k}/{len(lab_members)} to {lab_out}")
        for m in chosen:
            basename = Path(m).name
            out_path = lab_out / basename

            if args.skip_existing and out_path.exists():
                extracted_rows.append([lab, split, m, str(out_path), "skipped_exists"])
                continue

            try:
                data = zf.read(m, pwd=password.encode("utf-8"))
            except RuntimeError as e:
                raise RuntimeError(
                    "Failed to read a member from the zip (likely wrong password).\n"
                    f"Member: {m}\nError: {e}"
                )

            with open(out_path, "wb") as f:
                f.write(data)

            extracted_rows.append([lab, split, m, str(out_path), "ok"])

    zf.close()

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "split", "zip_member", "output_path", "status"])
        w.writerows(extracted_rows)

    print(f"[done] Wrote manifest: {manifest_path}")
    print(f"[done] Total extracted/recorded: {len(extracted_rows)}")
    print(f"[done] EcoSet zip stored at: {zip_path}")


if __name__ == "__main__":
    main()
