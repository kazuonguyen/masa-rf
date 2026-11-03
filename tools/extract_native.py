"""Batch MASA-R50 extraction with a strict checkpoint and explicit native backend."""
import argparse
import configparser
import csv
import hashlib
from pathlib import Path
import sys
import time
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from era_masa.data import read_ground_truth, save_cache, sha256
from era_masa.native_masa import NativeMasaR50
from era_masa.runlog import RunLog, run_directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default="MOT15")
    parser.add_argument("--data-root", type=Path, default=ROOT / "third_party/MASA-OA/benchmarks/MOT15")
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "saved_models/masa_r50.pth")
    parser.add_argument("--cache-root", type=Path, default=ROOT / "cache/MOT15")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    destination = args.output or run_directory(name="native-masa-extraction")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with RunLog(destination, config) as log:
        torch.set_num_threads(4)
        model = NativeMasaR50(args.checkpoint, ROOT / "third_party/MASA-OA", args.device)
        log.save("backend.json", {"backend": "original ResNet/FPN/DeformFusion; torchvision DCNv2+aligned ROIAlign",
            "loaded_checkpoint_keys": model.loaded_keys, "checkpoint_sha256": model.checkpoint_hash,
            "source_sha256": model.source_provenance, "torch": torch.__version__,
            "device": torch.cuda.get_device_name() if str(args.device).startswith("cuda") else args.device})
        log.message(f"Loaded {model.loaded_keys} MASA checkpoint tensors strictly")
        sequences = args.sequences or sorted(p.name for p in args.data_root.iterdir() if (p / "seqinfo.ini").exists())
        for name in sequences:
            source = args.data_root / name
            target = args.cache_root / f"{name}.npz"
            if target.exists():
                log.message(f"Existing cache preserved: {target}")
                continue
            gt = read_ground_truth(source, args.benchmark)
            info = configparser.ConfigParser()
            info.read(source / "seqinfo.ini")
            image_dir = source / info["Sequence"].get("imDir", "img1")
            extension = info["Sequence"].get("imExt", ".jpg")
            video = cv2.VideoCapture(str(source / f"{name}.mp4")) if not image_dir.is_dir() else None
            if video is not None and not video.isOpened():
                raise FileNotFoundError(f"Cannot open video for {name}")
            embeddings = np.empty((len(gt["frames"]), 256), dtype=np.float32)
            fingerprint = hashlib.sha256()
            elapsed_total = 0
            log.event("sequence_extraction_started", sequence=name, frames=gt["length"])
            try:
                with (destination / f"{name}-frames.csv").open("w", newline="", encoding="utf-8", buffering=1) as stream:
                    writer = csv.DictWriter(stream, fieldnames=["frame", "detections", "seconds"])
                    writer.writeheader()
                    for frame in range(1, gt["length"] + 1):
                        if video is not None:
                            ok, image = video.read()
                            if not ok:
                                raise RuntimeError(f"Video ended at {name}/{frame}")
                        else:
                            image = cv2.imread(str(image_dir / f"{frame:06d}{extension}"))
                            if image is None:
                                raise FileNotFoundError(f"Missing {name}/{frame}")
                        fingerprint.update(image.tobytes())
                        sl = slice(np.searchsorted(gt["frames"], frame, "left"), np.searchsorted(gt["frames"], frame, "right"))
                        if str(args.device).startswith("cuda"):
                            torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = model.extract(image, gt["boxes"][sl])
                        if str(args.device).startswith("cuda"):
                            torch.cuda.synchronize()
                        elapsed = time.perf_counter() - start
                        if not np.isfinite(output).all() or output.shape != (len(gt["boxes"][sl]), 256):
                            raise RuntimeError(f"Invalid MASA embedding at {name}/{frame}")
                        embeddings[sl] = output
                        elapsed_total += elapsed
                        writer.writerow({"frame": frame, "detections": len(output), "seconds": elapsed})
                        log.event("embedding_frame_completed", sequence=name, frame=frame, detections=len(output), seconds=elapsed)
                        if frame == 1 or frame % 50 == 0 or frame == gt["length"]:
                            log.message(f"{name}: {frame}/{gt['length']} frames; {frame/max(elapsed_total,1e-12):.2f} extraction FPS")
            finally:
                if video is not None:
                    video.release()
            metadata = {**gt["metadata"], "synthetic": False, "extractor": "MASA-R50-native-torchvision-operators",
                "checkpoint_sha256": model.checkpoint_hash, "loaded_keys": model.loaded_keys,
                "source_sha256": model.source_provenance, "decoded_images_sha256": fingerprint.hexdigest(),
                "extraction_seconds": elapsed_total, "extraction_fps": gt["length"] / max(elapsed_total, 1e-12)}
            save_cache(target, metadata=metadata, frames=gt["frames"], boxes=gt["boxes"], embeddings=embeddings,
                scores=np.ones(len(embeddings)), labels=np.zeros(len(embeddings), dtype=int),
                gt_ids=gt["gt_ids"], visibility=gt["visibility"])
            log.save(f"{name}-manifest.json", {**metadata, "cache_sha256": sha256(target)})
            log.message(f"Saved {target}")


if __name__ == "__main__":
    main()
