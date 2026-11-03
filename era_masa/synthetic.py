"""Deterministic diagnostic scenes. These are never benchmark results."""
from pathlib import Path
import numpy as np
from .data import save_cache


def create_synthetic(directory, seed=7):
    rng = np.random.default_rng(seed)
    directory = Path(directory)
    paths = []
    for name, corruption, empty in (("clean", False, False), ("crossing-poisoning", True, False),
                                    ("missing-frames", False, True)):
        features = np.linalg.qr(rng.normal(size=(256, 2)))[0].T
        frames, boxes, embeds, truth, visibility = [], [], [], [], []
        for frame in range(1, 81):
            if empty and 30 <= frame <= 33:
                continue
            for identity in range(2):
                x = 30 + 2 * frame if identity == 0 else 210 - 2 * frame
                y = 50 + identity * 4
                box = [x, y, x + 30, y + 70]
                emb = features[identity].copy() + rng.normal(0, 0.002, 256)
                visible = 1.0
                if corruption and 38 <= frame <= 51:
                    # Swap appearance while the trajectories retain velocity.
                    emb = 0.05 * features[identity] + 0.95 * features[1 - identity]
                    visible = 0.15
                frames.append(frame)
                boxes.append(box)
                embeds.append(emb * 8)  # Preserve embedding norm for bisoftmax.
                truth.append(identity + 1)
                visibility.append(visible)
        path = directory / f"{name}.npz"
        save_cache(path, metadata={"sequence": name, "length": 80, "fps": 30,
            "benchmark": "SYNTHETIC", "synthetic": True, "seed": seed, "extractor": "synthetic orthogonal vectors"},
            frames=np.array(frames), boxes=np.array(boxes), embeddings=np.array(embeds),
            scores=np.ones(len(frames)), labels=np.zeros(len(frames), dtype=int),
            gt_ids=np.array(truth), visibility=np.array(visibility))
        paths.append(path)
    return paths
