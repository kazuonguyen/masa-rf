"""Build TrackEval GT folders: full sequences or the ByteTrack/OC-SORT MOT17 val-half split.

val-half follows ByteTrack tools/convert_mot17_to_coco.py exactly: frames
n//2+2..n, renumbered by subtracting offset n//2+1.
"""
import argparse
import configparser
import json
import shutil
from pathlib import Path
import numpy as np


def half_offset(length):
    return length // 2 + 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "val-half", "train-half"), default="full")
    parser.add_argument("--only-frcnn", action="store_true")
    parser.add_argument("--sequences", nargs="+")
    args = parser.parse_args()
    seqinfo = {}
    for seq in sorted(p for p in args.split_root.iterdir() if (p / "seqinfo.ini").exists()):
        if args.only_frcnn and "FRCNN" not in seq.name:
            continue
        if args.sequences and seq.name not in args.sequences:
            continue
        info = configparser.ConfigParser()
        info.read(seq / "seqinfo.ini")
        length = int(info["Sequence"]["seqLength"])
        offset = half_offset(length) if args.mode == "val-half" else 0
        end = half_offset(length) if args.mode == "train-half" else length  # ByteTrack train_half: frames 1..n//2+1
        gt = np.loadtxt(seq / "gt" / "gt.txt", delimiter=",", ndmin=2)
        gt = gt[(gt[:, 0] > offset) & (gt[:, 0] <= end)]
        gt[:, 0] -= offset
        target = args.out / seq.name
        (target / "gt").mkdir(parents=True, exist_ok=True)
        fmt = ["%d", "%d"] + ["%.6g"] * (gt.shape[1] - 2)
        np.savetxt(target / "gt" / "gt.txt", gt, fmt=fmt, delimiter=",")
        info["Sequence"]["seqLength"] = str(end - offset)
        with (target / "seqinfo.ini").open("w") as stream:
            info.write(stream, space_around_delimiters=False)
        seqinfo[seq.name] = {"length": end - offset, "offset": offset, "source_length": length,
                             "fps": float(info["Sequence"]["frameRate"])}
    (args.out / "seqinfo.json").write_text(json.dumps({"mode": args.mode, "source": str(args.split_root.resolve()),
                                                       "sequences": seqinfo}, indent=2))
    print(f"{len(seqinfo)} sequences -> {args.out}")


if __name__ == "__main__":
    main()
