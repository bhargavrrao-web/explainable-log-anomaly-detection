"""Download ONLY the beginning of the public BGL log (Loghub, Zenodo) and keep the first N lines.

The full archive is 57 MB; we fetch just the first few MB with an HTTP Range request and decompress
the (deflate) stream from the start. A truncated zip is fine because the file is read sequentially.
Nothing from the archive is executed: we only inflate bytes and write text.

Run:  python src/get_bgl.py
Output: data/raw/BGL_200k.log   (path used by config.yaml -> paths.bgl_raw)
"""
import struct
import subprocess
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import load_config, p

URL = "https://zenodo.org/records/8196385/files/BGL.zip?download=1"
FETCH_BYTES = 15_000_000  # compressed bytes to fetch (plenty for 200k lines)


def main():
    cfg = load_config()
    n_lines = cfg["data"]["max_lines"]
    part = p("data/download_bgl/bgl_head.part")
    part.parent.mkdir(parents=True, exist_ok=True)
    if not part.exists():
        subprocess.run(["curl.exe", "-sL", "-m", "300", "-r", f"0-{FETCH_BYTES}", "-o", str(part), URL], check=True)
    raw = part.read_bytes()
    assert raw[:4] == b"PK\x03\x04", "not a zip local header"
    name_len, extra_len = struct.unpack("<HH", raw[26:30])
    method = struct.unpack("<H", raw[8:10])[0]
    assert method == 8, "expected deflate"
    data = raw[30 + name_len + extra_len:]
    d = zlib.decompressobj(-15)             # raw deflate stream
    out, lines, buf = [], 0, b""
    for i in range(0, len(data), 1 << 16):
        buf += d.decompress(data[i:i + (1 << 16)])
        *complete, buf = buf.split(b"\n")
        out.extend(complete)
        if len(out) >= n_lines:
            break
    out = out[:n_lines]
    dst = p("data/raw/BGL_200k.log")
    dst.write_bytes(b"\n".join(out) + b"\n")
    anom = sum(1 for l in out if not l.startswith(b"- "))
    print(f"wrote {dst.name}: {len(out)} lines, {anom} anomalous lines ({100 * anom / len(out):.1f}%)")


if __name__ == "__main__":
    main()
