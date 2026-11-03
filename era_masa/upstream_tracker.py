"""Execute the author's tracker source on cached MASA embeddings.

Only framework imports are replaced: BaseTracker.empty, a no-op registry,
typing and bbox IoU. Kalman and all association/update code run unmodified.
The native embedding backend is a separate, explicitly recorded substitution.
"""
import importlib.util
import sys
import types
from pathlib import Path
from typing import Any
import numpy as np
import torch
from mmengine.structures import InstanceData
from torchvision.ops import box_iou
from .data import sha256


class BaseTracker(torch.nn.Module):
    @property
    def empty(self):
        return not self.tracks


class Registry:
    def register_module(self):
        return lambda cls: cls


def load_author_tracker(root):
    root = Path(root).resolve() / "masa/models/tracker"
    prefix = "_era_author_tracker"
    if prefix + ".source" in sys.modules:
        return sys.modules[prefix + ".source"].MasaTaoTracker
    for name, folder in ((prefix, root), (prefix + ".MASA_OA_KF", root / "MASA_OA_KF")):
        module = types.ModuleType(name)
        module.__path__ = [str(folder)]
        sys.modules[name] = module
    path = root / "masa_tao_tracker.py"
    source = path.read_text(encoding="utf-8")
    for line in ("from mmdet.models.trackers.base_tracker import BaseTracker",
                 "from mmdet.registry import MODELS", "from mmdet.structures import TrackDataSample",
                 "from mmdet.structures.bbox import bbox_overlaps"):
        source = source.replace(line, "# Framework dependency supplied by cache adapter")
    module = types.ModuleType(prefix + ".source")
    module.__package__ = prefix
    module.__dict__.update(BaseTracker=BaseTracker, MODELS=Registry(), TrackDataSample=Any,
                           bbox_overlaps=box_iou)
    sys.modules[module.__name__] = module
    exec(compile(source, str(path), "exec"), module.__dict__)
    return module.MasaTaoTracker


class AuthorTracker:
    def __init__(self, config, root="third_party/MASA-OA"):
        self.model = load_author_tracker(root)(init_score_thr=config.init_score_threshold,
            obj_score_thr=config.object_score_threshold, match_score_thr=config.match_threshold,
            memo_tracklet_frames=config.max_age, memo_momentum=config.ema_alpha,
            max_distance=config.max_distance, fps=config.fps, with_cats=config.with_categories)
        self.diagnostics, self.pair_diagnostics = [], []
        self.statistics = {}
        self.tracks = self.model.tracks

    def update(self, frame, boxes, embeddings, scores, labels, diagnostics="matched"):
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        embeds = torch.as_tensor(embeddings, dtype=torch.float32)
        scores = torch.as_tensor(scores, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.long)
        pred = InstanceData(bboxes=boxes, scores=scores, labels=labels)
        sample = types.SimpleNamespace(pred_instances=pred,
                                      metainfo={"frame_id": frame - 1, "scale_factor": (1, 1)})
        head = types.SimpleNamespace(predict=lambda feats, rois: embeds)
        old_count = int(self.model.num_tracks)
        output = self.model.track(types.SimpleNamespace(track_head=head), None, [], sample)
        # Fixed valid GT scores exceed initialization threshold, so no box drops.
        if len(output.instances_id) != len(boxes):
            raise ValueError("Author cache adapter requires all detections retained")
        ids = np.empty(len(boxes), dtype=int)
        order = scores.sort(descending=True).indices.numpy()
        ids[order] = output.instances_id.numpy() + 1
        new_count = int(self.model.num_tracks) - old_count
        self.statistics = {"matches": len(ids) - new_count, "new_tracks": new_count,
                           "frozen_updates": 0}
        self.diagnostics, self.pair_diagnostics = [], []
        return ids


def source_provenance(root):
    folder = Path(root) / "masa/models/tracker"
    return {str(p.relative_to(folder)): sha256(p) for p in
            (folder / "masa_tao_tracker.py", folder / "MASA_OA_KF/kalman_filter.py",
             folder / "MASA_OA_KF/track.py")}
