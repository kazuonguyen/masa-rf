"""YOLOX (official ByteTrack/DanceTrack checkpoints) -> MASA-R50 embeddings -> camera motion.

One pass per sequence writes a schema-2 detector cache. Images are decoded once;
GT is never read here. Timings (detector / MASA / GMC) are logged per frame.
"""
import argparse
import configparser
import csv
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party/ByteTrack"))
from era_masa.data import sha256
from era_masa.detcache import save_det_cache
from era_masa.native_masa import NativeMasaR50
from era_masa.runlog import RunLog, run_directory

DETECTORS = {
    # name: (exp file, checkpoint, test_size)
    "mot17-ablation": ("exps/example/mot/yolox_x_ablation.py", "saved_models/yolox/bytetrack_ablation.pth.tar"),
    "mot17": ("exps/example/mot/yolox_x_mix_det.py", "saved_models/yolox/bytetrack_x_mot17.pth.tar"),
    "mot20": ("exps/example/mot/yolox_x_mix_mot20_ch.py", "saved_models/yolox/bytetrack_x_mot20.pth.tar"),
    "dancetrack": ("exps/example/mot/yolox_x_ablation.py", "saved_models/yolox/dancetrack_bytetrack_model.pth.tar"),
}
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class Detector:
    def __init__(self, name, device):
        from yolox.exp import get_exp
        from yolox.utils import postprocess
        exp_file, ckpt = DETECTORS[name]
        self.exp = get_exp(str(ROOT / "third_party/ByteTrack" / exp_file), None)
        self.model = self.exp.get_model()
        state = torch.load(ROOT / ckpt, map_location="cpu", weights_only=False)["model"]
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(f"Detector checkpoint mismatch: {missing[:3]} {unexpected[:3]}")
        self.model.to(device).eval().half()
        self.postprocess = postprocess
        self.device = device
        self.size = self.exp.test_size
        self.nms = self.exp.nmsthre
        self.checkpoint_sha256 = sha256(ROOT / ckpt)
        self.exp_file = exp_file

    @torch.inference_mode()
    def __call__(self, bgr, conf):
        h, w = bgr.shape[:2]
        r = min(self.size[0] / h, self.size[1] / w)
        resized = cv2.resize(bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).to(self.device)
        padded = torch.full((self.size[0], self.size[1], 3), 114.0, device=self.device)
        padded[: resized.shape[0], : resized.shape[1]] = tensor.float()
        x = padded[:, :, [2, 1, 0]] / 255.0  # BGR -> RGB exactly as ByteTrack preproc
        x = (x - x.new_tensor(MEAN)) / x.new_tensor(STD)
        x = x.permute(2, 0, 1)[None].half()
        out = self.postprocess(self.model(x), 1, conf, self.nms)[0]
        if out is None:
            return np.empty((0, 4)), np.empty(0)
        out = out.float().cpu().numpy()
        boxes = out[:, :4] / r
        scores = out[:, 4] * out[:, 5]
        return boxes, scores


class SparseGMC:
    """BoT-SORT 'sparseOptFlow' global motion compensation (affine, partial)."""

    def __init__(self, downscale=2):
        self.downscale = downscale
        self.prev = None
        self.prev_points = None

    def __call__(self, bgr):
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        gray = cv2.resize(gray, (w // self.downscale, h // self.downscale))
        H = np.eye(2, 3)
        points = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=1,
                                         blockSize=3, useHarrisDetector=False, k=0.04)
        if self.prev is not None and self.prev_points is not None and len(self.prev_points) > 4:
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, gray, self.prev_points, None)
            ok = status.ravel() == 1
            if ok.sum() > 4:
                A, _ = cv2.estimateAffinePartial2D(self.prev_points[ok], nxt[ok], method=cv2.RANSAC)
                if A is not None:
                    H = A
                    H[:, 2] *= self.downscale
        self.prev, self.prev_points = gray, points
        return H


def sequences_of(split_root):
    return sorted(p for p in Path(split_root).iterdir() if (p / "seqinfo.ini").exists())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)  # MOT17, MOT20, DanceTrack
    parser.add_argument("--split-root", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--detector", choices=sorted(DETECTORS), required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--sequences", nargs="+")
    parser.add_argument("--conf", type=float, default=0.05)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "saved_models/masa_r50.pth")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    destination = args.output or run_directory(name=f"extract-{args.benchmark}-{args.split}-{args.detector}")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    with RunLog(destination, config) as log:
        detector = Detector(args.detector, args.device)
        masa = NativeMasaR50(args.checkpoint, ROOT / "third_party/MASA-OA", args.device)
        log.save("backend.json", {"detector": args.detector, "detector_exp": detector.exp_file,
            "detector_sha256": detector.checkpoint_sha256, "test_size": detector.size, "nms": detector.nms,
            "conf_floor": args.conf, "masa_sha256": masa.checkpoint_hash, "masa_keys": masa.loaded_keys,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name()})
        seqs = sequences_of(args.split_root)
        if args.sequences:
            seqs = [s for s in seqs if s.name in args.sequences]
        for source in seqs:
            target = args.cache_root / f"{source.name}.npz"
            if target.exists():
                log.message(f"Existing cache preserved: {target}")
                continue
            info = configparser.ConfigParser()
            info.read(source / "seqinfo.ini")
            sec = info["Sequence"]
            length, ext = int(sec["seqLength"]), sec.get("imExt", ".jpg")
            if args.max_frames:
                length = min(length, args.max_frames)
            image_dir = source / sec.get("imDir", "img1")
            gmc_model = SparseGMC()
            frames, boxes, scores, embeds = [], [], [], []
            gmc = np.zeros((length, 2, 3))
            fingerprint = hashlib.sha256()
            timing = np.zeros(3)
            width = height = None
            with (destination / f"{source.name}-frames.csv").open("w", newline="", buffering=1) as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame", "detections", "detector_s", "masa_s", "gmc_s"])
                def read(frame):
                    image = cv2.imread(str(image_dir / f"{frame:06d}{ext}"))
                    return image if image is not None else cv2.imread(str(image_dir / f"{frame:08d}{ext}"))

                # Decode ahead on worker threads (cv2 releases the GIL); order is preserved.
                pool = ThreadPoolExecutor(4)
                pending = {f: pool.submit(read, f) for f in range(1, min(length, 8) + 1)}
                for frame in range(1, length + 1):
                    image = pending.pop(frame).result()
                    if frame + 8 <= length:
                        pending[frame + 8] = pool.submit(read, frame + 8)
                    if image is None:
                        raise FileNotFoundError(f"Missing image {source.name}/{frame}")
                    height, width = image.shape[:2]
                    fingerprint.update(np.ascontiguousarray(image[::16, ::16]).tobytes())
                    t0 = time.perf_counter()
                    b, s = detector(image, args.conf)
                    torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    # Trackers and evaluation use the UNCLIPPED detector boxes (as ByteTrack does; MOT GT
                    # extends beyond the frame). Only MASA's RoI pooling uses the in-image part.
                    c = b.copy()
                    c[:, [0, 2]] = c[:, [0, 2]].clip(0, width - 1)
                    c[:, [1, 3]] = c[:, [1, 3]].clip(0, height - 1)
                    keep = (c[:, 2] - c[:, 0] > 1) & (c[:, 3] - c[:, 1] > 1) & (s >= args.conf)
                    b, c, s = b[keep], c[keep], s[keep]
                    e = masa.extract(image, c)
                    torch.cuda.synchronize()
                    t2 = time.perf_counter()
                    gmc[frame - 1] = gmc_model(image)
                    t3 = time.perf_counter()
                    timing += (t1 - t0, t2 - t1, t3 - t2)
                    if not np.isfinite(e).all():
                        raise RuntimeError(f"Nonfinite MASA embedding {source.name}/{frame}")
                    frames.append(np.full(len(b), frame)); boxes.append(b); scores.append(s); embeds.append(e)
                    writer.writerow([frame, len(b), t1 - t0, t2 - t1, t3 - t2])
                    if frame % 200 == 0 or frame == length:
                        log.message(f"{source.name}: {frame}/{length} | {frame / timing.sum():.1f} FPS "
                                    f"(det {frame / timing[0]:.1f}, masa {frame / timing[1]:.1f}, gmc {frame / timing[2]:.1f})")
            meta = {"sequence": source.name, "benchmark": args.benchmark, "split": args.split,
                    "length": length, "fps": float(sec["frameRate"]), "width": width, "height": height,
                    "detector": args.detector, "detector_sha256": detector.checkpoint_sha256,
                    "masa_sha256": masa.checkpoint_hash, "conf_floor": args.conf,
                    "image_fingerprint": fingerprint.hexdigest(),
                    "seconds": {"detector": timing[0], "masa": timing[1], "gmc": timing[2]}}
            save_det_cache(target, metadata=meta, frames=np.concatenate(frames), boxes=np.concatenate(boxes),
                           scores=np.concatenate(scores), embeddings=np.concatenate(embeds), gmc=gmc)
            log.save(f"{source.name}-manifest.json", {**meta, "cache_sha256": sha256(target)})
            log.message(f"Saved {target}")


if __name__ == "__main__":
    main()
