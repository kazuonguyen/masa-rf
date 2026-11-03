# MASA-RF: mã nguồn của report

Repo này gồm [report](report.pdf), phần code tạo ra kết quả trong report và [log mẫu](logs/README.txt) của một số lần chạy. Không kèm dataset, trọng số mô hình, cache đặc trưng và mã nguồn bên thứ ba.

## Cấu trúc

| Thư mục, file | Nội dung |
|---|---|
| `era_masa/det_tracker.py` | **MASA-RF**, phương pháp đề xuất |
| `era_masa/det_baselines.py` | Adapter cho các baseline: ByteTrack, OC-SORT, BoT-SORT, Deep OC-SORT, MASA, MASA-OA |
| `era_masa/det_run.py` | Chạy một tracker trên một split và chấm bằng TrackEval |
| `era_masa/` (còn lại) | Đọc cache, ghi log, Kalman, thí nghiệm với box ground truth |
| `tools/` | Trích đặc trưng, dò tham số, chạy thí nghiệm, thống kê |
| `configs/det_locked.json` | Cấu hình đã khóa của mọi phương pháp |
| `configs/det_tuning_spaces.json` | Không gian dò tham số |
| `tests/` | Kiểm thử tự động |

## Cài đặt

Môi trường thí nghiệm: Windows 11, Python 3.12, NVIDIA RTX 5070 Ti.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r configs/requirements-cpu.lock.txt
.venv/Scripts/python.exe -m pip install -e ".[test]"

python -m venv .venv-gpu
.venv-gpu/Scripts/python.exe -m pip install -r configs/requirements-gpu.lock.txt
.venv-gpu/Scripts/python.exe -m pip install -e ".[test]"
.venv-gpu/Scripts/python.exe -m pytest -q tests
```

Cần tải thêm:

| Thành phần | Vị trí | Nguồn |
|---|---|---|
| ByteTrack, OC_SORT, BoT-SORT, Deep-OC-SORT, MASA-OA, TrackEval | `third_party/` | Đúng commit trong `configs/third_party_versions.json` |
| MOT15 | `third_party/MASA-OA/benchmarks/MOT15` | Có sẵn trong repo MASA-OA |
| MOT17, MOT20 | `data/MOT17-full`, `data/MOT20-full` | https://motchallenge.net |
| DanceTrack | `data/DanceTrack` | https://github.com/DanceTrack/DanceTrack |
| YOLOX-X | `saved_models/yolox/` | Repo ByteTrack và DanceTrack |
| MASA-R50 | `saved_models/masa_r50.pth` | https://github.com/siyuanliii/masa |

## Chạy

Các lệnh dưới đây dùng `PY=.venv-gpu/Scripts/python.exe`.

**1. Ground truth**

```bash
MOT=data/MOT17-full/MOT17/train
TUNE="dancetrack0001 dancetrack0002 dancetrack0006 dancetrack0008 dancetrack0012 dancetrack0015"
$PY tools/prepare_gt.py --split-root $MOT --out gt/MOT17-val-half --mode val-half --only-frcnn
$PY tools/prepare_gt.py --split-root $MOT --out gt/MOT17-train-half --mode train-half --only-frcnn
$PY tools/prepare_gt.py --split-root data/MOT20-full/MOT20/train --out gt/MOT20-train
$PY tools/prepare_gt.py --split-root data/DanceTrack/val --out gt/DanceTrack-val
$PY tools/prepare_gt.py --split-root data/DanceTrack/train1 --out gt/DanceTrack-train-tune --sequences $TUNE
```

**2. Trích detection, embedding MASA và chuyển động camera** (cần GPU)

```bash
F17="MOT17-02-FRCNN MOT17-04-FRCNN MOT17-05-FRCNN MOT17-09-FRCNN MOT17-10-FRCNN MOT17-11-FRCNN MOT17-13-FRCNN"
$PY tools/extract_detections.py --benchmark MOT17 --split-root $MOT --split train --detector mot17-ablation --cache-root cache_det/MOT17-train-ablation --sequences $F17
$PY tools/extract_detections.py --benchmark MOT20 --split-root data/MOT20-full/MOT20/train --split train --detector mot17 --cache-root cache_det/MOT20-train-mot17det
$PY tools/extract_detections.py --benchmark DanceTrack --split-root data/DanceTrack/val --split val --detector dancetrack --cache-root cache_det/DanceTrack-val
$PY tools/extract_detections.py --benchmark DanceTrack --split-root data/DanceTrack/train1 --split train --detector dancetrack --cache-root cache_det/DanceTrack-train --sequences $TUNE
$PY tools/restore_unclipped_boxes.py --src cache_det/MOT17-train-ablation cache_det/MOT20-train-mot17det cache_det/DanceTrack-val cache_det/DanceTrack-train --output outputs/restore-unclipped
```

**3. Dò tham số trên tập tuning, khóa cấu hình, chạy trên tập báo cáo**

```bash
bash tools/run_v2_pipeline.sh
```

Script tự bỏ qua bước tạo file nộp test server nếu chưa có cache của tập test.

**4. Thí nghiệm với box ground truth trên MOT15**

```bash
$PY tools/extract_native.py
$PY tools/ablate_author_oa.py
$PY tools/evaluate_published.py
for c in geometry_only iou_kf_only appearance_only; do
  $PY -m era_masa run --cache cache/MOT15/*.npz --config configs/gtbox_$c.json --variants robust-fusion --evaluate --no-subsets --diagnostics none --output outputs/gtbox-baselines/gtbox_$c-MOT15
done
```

**5. Thí nghiệm bổ sung, đo thời gian, thống kê** (chạy sau bước 4)

```bash
$PY tools/run_paper_experiments.py
$PY tools/time_pipeline.py
$PY tools/paper_analysis.py
```

Chạy riêng MASA-RF trên một split:

```bash
.venv-gpu/Scripts/python.exe -m era_masa.det_run --method masa-rf --params "<params trong configs/det_locked.json>" --caches cache_det_v2/DanceTrack-val --gt-root gt/DanceTrack-val --benchmark DanceTrack --output outputs/masa-rf-dancetrack
```

Mỗi lần chạy ghi ra `console.log`, `config.json`, `metrics.csv` và kết quả tracker theo định dạng MOTChallenge trong thư mục `--output`.
