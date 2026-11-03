"""Mechanism ablation of the authors' MASA-OA tracker source (fixed GT boxes, MOT15).

Each variant is a textual patch of a *copy* of the authors' file; the original in
third_party is never modified. Patched sources and their SHA256 are kept.

  original     : unmodified source
  kf-once      : final per-frame update keeps EMA but skips the second Kalman correction
  freeze       : final update no longer touches embeddings of existing tracks (paper's alpha=0 honoured)
  single       : final update only initialises new tracks (paper semantics: freeze + one KF correction)
  thr03        : overlap clustering uses IoU>=0.3 as written in the paper (code uses IoU>0)
  single-thr03 : both paper semantics
"""
import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FINAL = "        self.update(ids, bboxes, embeds, labels, scores, frame_id)\n        tracklet_inds"
VARIANTS = ["original", "kf-once", "freeze", "single", "thr03", "single-thr03"]


def patch(src, name):
    assert src.count(FINAL) == 1
    if name in ("single", "single-thr03"):
        src = src.replace(FINAL, "        self.update(torch.where(new_inds.to(ids.device), ids, torch.full_like(ids, -1)), "
                          "bboxes, embeds, labels, scores, frame_id)\n        tracklet_inds")
    if name == "freeze":
        src = src.replace(FINAL, "        self.update(ids, bboxes, embeds, labels, scores, frame_id, momentum=0)\n        tracklet_inds")
    if name == "kf-once":
        line = "if self.use_kf and info.get('dstrack'):"
        assert src.count(line) == 1
        src = src.replace(line, "if self.use_kf and info.get('dstrack') and not getattr(self, '_skip_kf', False):")
        src = src.replace(FINAL, "        self._skip_kf = True\n        self.update(ids, bboxes, embeds, labels, scores, frame_id)\n"
                                 "        self._skip_kf = False\n        tracklet_inds")
    if name in ("thr03", "single-thr03"):
        assert src.count("(iou_mat > 0).float()") == 1 and src.count("if iou_mat[i,j]>0:") == 1
        src = src.replace("(iou_mat > 0).float()", "(iou_mat >= 0.3).float()").replace("if iou_mat[i,j]>0:", "if iou_mat[i,j]>=0.3:")
    return src


def child(root, output):
    sys.path.insert(0, str(ROOT))
    import era_masa.upstream_tracker as ut
    ut.AuthorTracker.__init__.__defaults__ = (str(root),)
    from era_masa.cli import main
    caches = sorted(str(x) for x in (ROOT / "cache/MOT15").glob("*.npz"))
    main(["run", "--cache", *caches, "--variants", "author-oa", "--evaluate", "--no-subsets",
          "--diagnostics", "none", "--output", str(output)])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=ROOT / "outputs/mot15-author-mechanism-ablation")
    p.add_argument("--child", nargs=2)
    a = p.parse_args()
    if a.child:
        return child(Path(a.child[0]), Path(a.child[1]))
    a.output.mkdir(parents=True, exist_ok=False)
    original = ROOT / "third_party/MASA-OA/masa/models/tracker"
    for name in VARIANTS:
        root = a.output / "sources" / name
        shutil.copytree(original, root / "masa/models/tracker")
        f = root / "masa/models/tracker/masa_tao_tracker.py"
        f.write_text(patch(f.read_text(encoding="utf-8"), name) if name != "original" else f.read_text(encoding="utf-8"),
                     encoding="utf-8")
        subprocess.run([sys.executable, __file__, "--child", str(root), str(a.output / "runs" / name)], check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
