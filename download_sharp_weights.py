"""Download only the SHARP checkpoint used by the workbench.

Run from any directory: python /path/to/sharp-video/download_sharp_weights.py
Uses Python's standard library; no PyTorch or other packages are required.
An existing checkpoint is left untouched. Interrupted downloads remain in a
separate .tmp file and restart on the next invocation.
"""

from pathlib import Path
import sys
from urllib.request import urlretrieve


# Official URL from ml-sharp/src/sharp/cli/predict.py (DEFAULT_MODEL_URL).
MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"
CHECKPOINT = Path(__file__).resolve().parent / "ml-sharp" / "checkpoints" / "sharp_2572gikvuh.pt"


def download_checkpoint(destination: Path = CHECKPOINT) -> Path:
    destination = Path(destination)
    if destination.is_file():
        print(f"Already exists; keeping: {destination}", flush=True)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    print(f"Downloading SHARP from {MODEL_URL}", flush=True)
    print(f"Destination: {destination}", flush=True)

    def progress(blocks: int, block_size: int, total: int) -> None:
        if not sys.stdout.isatty():
            return
        received = blocks * block_size
        if total > 0:
            received = min(received, total)
            message = f"{received / 1048576:.1f} / {total / 1048576:.1f} MiB ({received / total:.1%})"
        else:
            message = f"{received / 1048576:.1f} MiB"
        print(f"\r{message}", end="", flush=True)

    # urlretrieve rejects a download shorter than the advertised Content-Length.
    # Only publish the final filename after the entire transfer succeeds.
    urlretrieve(MODEL_URL, temporary, reporthook=progress)
    temporary.replace(destination)
    print(f"\nSaved: {destination} ({destination.stat().st_size:,} bytes)", flush=True)
    return destination


if __name__ == "__main__":
    try:
        download_checkpoint()
    except KeyboardInterrupt:
        print("\nDownload interrupted; the existing checkpoint was not changed.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        print(f"Download failed: {error}", file=sys.stderr)
        sys.exit(1)
