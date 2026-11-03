"""Extract real MASA-R50 embeddings, bypassing upstream association entirely."""
import configparser
import csv
import hashlib
import sys
import time
from pathlib import Path
import numpy as np
from .data import read_ground_truth, save_cache, sha256


def extract_sequence(args, log):
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"MASA checkpoint missing: {args.checkpoint}")
    if args.cache_output.exists():
        raise FileExistsError(f"Cache already exists: {args.cache_output}")
    root = args.masa_root.resolve()
    sys.path.insert(0, str(root))
    try:
        import torch
        import cv2
        import masa  # registers upstream model components
        from masa.apis import init_masa, inference_masa, build_test_pipeline
        from mmengine.structures import InstanceData
    except ImportError as exc:
        raise RuntimeError("Real embedding extraction requires the MASA/OpenMMLab environment. "
                           "See docs/EXPERIMENTS.md; the lightweight tracker environment is separate.") from exc
    config = args.masa_config or root / "configs/masa-one/masa_r50_plug_and_play.py"
    # Some upstream relative paths resolve from cwd. Config paths are absolute.
    gt = read_ground_truth(args.sequence, args.benchmark, args.gt_policy)
    model = init_masa(str(config.resolve()), str(args.checkpoint.resolve()), device=args.device)
    pipeline = build_test_pipeline(model.cfg)

    class CaptureTracker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.captured = None

        def reset(self):
            self.captured = None

        def track(self, model, img, feats, data_sample, rescale=True, **kwargs):
            instances = data_sample.pred_instances
            boxes = instances.bboxes
            if len(boxes):
                scaled = boxes.clone()
                if rescale:
                    scaled *= scaled.new_tensor(data_sample.metainfo["scale_factor"]).repeat(1, 2)
                self.captured = model.track_head.predict(feats, [scaled]).detach().cpu().numpy()
            else:
                self.captured = np.empty((0, 256), dtype=np.float32)
            result = instances.clone()
            result.instances_id = torch.arange(len(boxes), device=boxes.device)
            result.mask_inds = torch.arange(len(boxes), device=boxes.device)
            return result

    capture = CaptureTracker()
    model.tracker = capture
    info = configparser.ConfigParser()
    info.read(args.sequence / "seqinfo.ini")
    image_dir = args.sequence / info["Sequence"].get("imDir", "img1")
    extension = info["Sequence"].get("imExt", ".jpg")
    video_path = args.sequence / f"{args.sequence.name}.mp4"
    video = None
    if not image_dir.is_dir():
        video = cv2.VideoCapture(str(video_path))
        if not video.isOpened():
            raise FileNotFoundError(f"Neither {image_dir} nor readable {video_path} found")
    images_digest = hashlib.sha256()
    features = np.empty((len(gt["frames"]), 256), dtype=np.float32)
    times = []
    try:
        with (log.directory / "extraction_frames.csv").open("w", newline="", encoding="utf-8", buffering=1) as stream:
            writer = csv.DictWriter(stream, fieldnames=["frame", "detections", "embedding_seconds", "embedding_fps"])
            writer.writeheader()
            for frame in range(1, gt["length"] + 1):
                if video is not None:
                    success, image = video.read()
                    if not success:
                        raise RuntimeError(f"Video ended before frame {frame}")
                else:
                    image = cv2.imread(str(image_dir / f"{frame:06d}{extension}"))
                    if image is None:
                        raise FileNotFoundError(f"Missing image at frame {frame}")
                images_digest.update(image.tobytes())
                sl = slice(np.searchsorted(gt["frames"], frame, "left"), np.searchsorted(gt["frames"], frame, "right"))
                boxes = gt["boxes"][sl]
                # GT boxes are fixed; classification/confidence does not measure
                # embedding quality. Conf=1 for every retained annotation.
                det_boxes = torch.as_tensor(np.column_stack([boxes, np.ones(len(boxes))]),
                                            dtype=torch.float32, device=args.device)
                det_labels = torch.zeros(len(boxes), dtype=torch.long, device=args.device)
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                capture.captured = None
                inference_masa(model, image, frame_id=frame - 1, video_len=gt["length"],
                    test_pipeline=pipeline, det_bboxes=det_boxes, det_labels=det_labels)
                if str(args.device).startswith("cuda"):
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                if capture.captured is None or capture.captured.shape != (len(boxes), 256):
                    raise RuntimeError("MASA extraction returned misaligned embeddings")
                features[sl] = capture.captured
                times.append(elapsed)
                writer.writerow({"frame": frame, "detections": len(boxes), "embedding_seconds": elapsed,
                                 "embedding_fps": 1 / max(elapsed, 1e-12)})
                log.event("embedding_frame_completed", frame=frame, detections=len(boxes), seconds=elapsed)
                if frame == 1 or frame % 100 == 0 or frame == gt["length"]:
                    log.message(f"MASA extraction | {args.sequence.name} | {frame}/{gt['length']} | {1/max(elapsed,1e-12):.2f} FPS")
    finally:
        if video is not None:
            video.release()
    metadata = {**gt["metadata"], "synthetic": False, "extractor": "MASA-R50/track_head.predict",
        "checkpoint_sha256": sha256(args.checkpoint), "masa_config_sha256": sha256(config),
        "decoded_images_sha256": images_digest.hexdigest(), "device": args.device,
        "extraction_seconds": sum(times), "extraction_fps": gt["length"] / max(sum(times), 1e-12)}
    save_cache(args.cache_output, metadata=metadata, frames=gt["frames"], boxes=gt["boxes"],
               embeddings=features, scores=np.ones(len(features)), labels=np.zeros(len(features), dtype=int),
               gt_ids=gt["gt_ids"], visibility=gt["visibility"])
    log.save("cache_manifest.json", {**metadata, "cache": str(args.cache_output.resolve()),
                                    "cache_sha256": sha256(args.cache_output)})
    log.message(f"Saved real MASA embeddings: {args.cache_output.resolve()}")
