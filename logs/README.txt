Log mẫu của MASA-RF. Các file được sao chép nguyên từ lần chạy thật, không chỉnh sửa.

Lệnh tạo ra một lần chạy (ví dụ trên DanceTrack-val; cần có dữ liệu và cache nên không kèm ở đây):
  python -m era_masa.det_run --method masa-rf --params "<params trong config_DanceTrack-val.json>"
    --caches cache_det_v2/DanceTrack-val --gt-root gt/DanceTrack-val --benchmark DanceTrack
    --output outputs/det-suite/DanceTrack-val/masa-rf

console_MOT17-val.log, console_MOT20-val.log, console_DanceTrack-val.log
  Nội dung code in ra khi chạy MASA-RF: mỗi video một dòng (số box, số ID, tốc độ), hai dòng cuối là kết quả cả tập.
  Dòng "masa-rf" là số trong report; dòng "masa-rf-interp" là kết quả khi có nội suy.
  Tốc độ FPS ở đây đo lúc nhiều lần chạy song song, nên không phải số trong report.

config_DanceTrack-val.json
  Tham số của lần chạy MASA-RF trên DanceTrack-val.

metrics_DanceTrack-val.csv
  Kết quả TrackEval của MASA-RF theo từng video (HOTA, AssA, DetA, IDF1, IDSW...).

summary_DanceTrack-val.csv
  Kết quả của mọi phương pháp trên DanceTrack-val.
  Hậu tố -interp: có nội suy. -default: tham số mặc định. no-...: bỏ một thành phần của MASA-RF.

timing_association.json
  Thời gian chạy phần ghép cặp, đo lại từng phương pháp một trên máy rảnh, 3 lần; đây là số dùng trong report.
  fps_median: FPS trung vị. ms_per_frame: mili giây mỗi frame.
  identical_to_suite: true nghĩa là kết quả tracker giống hệt từng byte với lần chạy chính.
