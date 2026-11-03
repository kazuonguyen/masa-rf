from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path

BASELINES = ("masa", "masa-hung", "masa-hung-kf", "masa-clust2", "masa-clust8", "masa-oa")
ABLATIONS = ("soft-iou", "temporal", "temporal-rival", "ecr-association", "memory-only", "era-full")
VARIANTS = BASELINES + ABLATIONS
EXTRA_VARIANTS = ("author-oa", "hybrid", "cascade", "gallery", "observation", "robust-fusion")
SUPPORTED_VARIANTS = VARIANTS + EXTRA_VARIANTS


@dataclass
class TrackerConfig:
    variant: str = "era-full"
    max_age: int = 10
    fps: float = 30.0
    max_distance: float = 100.0
    match_threshold: float = 0.5
    iou_threshold: float = 0.3
    overlap_low: float = 0.3
    overlap_high: float = 0.7
    init_score_threshold: float = 0.1
    object_score_threshold: float = 0.01
    ema_alpha: float = 0.8
    fallback_alpha: float = 0.35
    slow_alpha: float = 0.1
    reliability_low: float = 0.35
    reliability_high: float = 0.75
    margin_threshold: float = 0.05
    rival_margin: float = 0.2
    neighbor_radius: float = 2.0
    motion_scale: float = 9.4877
    motion_enabled: bool = True
    geometry_motion_mix: float = 0.5
    appearance_weight: float = 0.7
    gallery_size: int = 5
    direction_weight: float = 0.0
    velocity_alpha: float = 1.0
    observation_mix: float = 1.0
    center_weight: float = 0.0
    adaptive_appearance: float = 0.0
    memory_risk_factor: float = 0.0
    age_penalty: float = 0.0
    age_appearance_gate: float = 0.0
    camera_motion: bool = False
    with_categories: bool = False
    weights: dict = field(default_factory=lambda: {
        "temporal": 0.30, "rival": 0.35, "geometry": 0.20, "motion": 0.15})

    def __post_init__(self):
        for key, value in asdict(self).items():
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite")
        if self.variant not in SUPPORTED_VARIANTS:
            raise ValueError(f"Unknown variant {self.variant}; choose from {SUPPORTED_VARIANTS}")
        if self.max_age < 1 or self.fps <= 0:
            raise ValueError("max_age must be positive; fps must be positive")
        if self.max_distance != -1 and self.max_distance <= 0:
            raise ValueError("max_distance must be positive or -1 (disabled)")
        for key in ("match_threshold", "iou_threshold", "overlap_low", "overlap_high",
                    "ema_alpha", "fallback_alpha", "slow_alpha", "reliability_low",
                    "reliability_high", "margin_threshold", "geometry_motion_mix",
                    "init_score_threshold", "object_score_threshold"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if not 0 <= self.appearance_weight <= 1 or not 0 <= self.direction_weight <= 1:
            raise ValueError("Appearance and direction weights must be in [0, 1]")
        for key in ("velocity_alpha", "observation_mix", "center_weight", "adaptive_appearance", "memory_risk_factor", "age_penalty", "age_appearance_gate"):
            if not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0, 1]")
        if self.gallery_size < 1 or int(self.gallery_size) != self.gallery_size:
            raise ValueError("Gallery size must be a positive integer")
        if not self.overlap_low < self.overlap_high:
            raise ValueError("overlap_low must be below overlap_high")
        if not self.reliability_low < self.reliability_high:
            raise ValueError("reliability_low must be below reliability_high")
        if self.motion_scale <= 0 or self.rival_margin <= 0 or self.neighbor_radius <= 0:
            raise ValueError("Signal scales must be positive")
        if set(self.weights) != {"temporal", "rival", "geometry", "motion"}:
            raise ValueError("weights must contain temporal, rival, geometry, motion")
        if any(not math.isfinite(v) or v < 0 for v in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("ECR weights must be nonnegative with positive sum")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None = None, **overrides):
        values = json.loads(Path(path).read_text(encoding="utf-8")) if path else {}
        values.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**values)
