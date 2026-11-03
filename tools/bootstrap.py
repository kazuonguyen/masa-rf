"""Fetch pinned source repositories, without installing GPU dependencies."""
import argparse
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "MASA-OA": ("https://github.com/gonzalocarreteroh/MASA-OA.git", "2dc7305b0d584aee7df45d83ed3c42804fa055fd"),
    "TrackEval": ("https://github.com/JonathonLuiten/TrackEval.git", "12c8791b303e0a0b50f753af204249e622d0281a"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--git", default=shutil.which("git"))
    args = parser.parse_args()
    if not args.git:
        parser.error("git executable missing; pass --git PATH")
    for name, (url, commit) in SOURCES.items():
        destination = ROOT / "third_party" / name
        if not destination.exists():
            subprocess.run([args.git, "clone", url, str(destination)], check=True)
            if commit:
                subprocess.run([args.git, "-C", str(destination), "checkout", "--detach", commit], check=True)
        head = subprocess.check_output([args.git, "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
        if commit and head != commit:
            raise RuntimeError(f"Existing {destination} is at {head}; expected {commit}. Preserve it and use a separate checkout.")
        print(f"{name}: {head}")


if __name__ == "__main__":
    main()
