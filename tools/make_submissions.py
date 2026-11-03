"""Produce test-server submission zips from LOCKED configs (no GT involved).

MOT17: the same tracker output is copied to the DPM/FRCNN/SDP names (all 21 files),
       as done by ByteTrack/OC-SORT/BoT-SORT submissions on private-detection track.
MOT20, DanceTrack: one file per test sequence.
Both raw and linearly interpolated (gap<=20) variants are written. Uploading requires
the user's MOTChallenge / CodaLab account.
"""
import argparse
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv-gpu/Scripts/python.exe")
TESTS = {"MOT17": ("cache_det_v2/MOT17-test", "mot"), "MOT20": ("cache_det_v2/MOT20-test", "mot"),
         "DanceTrack": ("cache_det_v2/DanceTrack-test", "dancetrack")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--methods", nargs="+", default=["deepocsort-rf", "masa-rf"])
    p.add_argument("--output", type=Path, default=ROOT / "outputs/submissions")
    a = p.parse_args()
    locked = json.loads((ROOT / "configs/det_locked.json").read_text())
    for bench, (caches, fam) in TESTS.items():
        for m in a.methods:
            entry = next(e for e in locked[fam] if e["name"] == m)
            run = a.output / bench / m
            if not (run / "trackers").exists():
                subprocess.run([PY, "-m", "era_masa.det_run", "--method", entry["method"], "--params",
                                json.dumps(entry["params"]), "--name", m, "--caches", caches,
                                "--benchmark", bench, "--output", str(run)], cwd=ROOT, check=True)
            for variant in (m, f"{m}-interp"):
                data = run / "trackers" / variant / "data"
                z = a.output / f"{bench}-{variant}.zip"
                with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
                    for f in sorted(data.glob("*.txt")):
                        if bench == "MOT17":
                            for det in ("DPM", "FRCNN", "SDP"):
                                zf.write(f, f.name.replace("FRCNN", det))
                        else:
                            # DanceTrack (CodaBench) expects the txt files inside a folder named "tracker"
                            zf.write(f, f"tracker/{f.name}" if bench == "DanceTrack" else f.name)
                print(z, len(zipfile.ZipFile(z).namelist()), "files")


if __name__ == "__main__":
    main()
