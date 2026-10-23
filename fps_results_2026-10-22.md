# 2560x1440 in implication space — measured results, 2026-10-22

Frame: synthetic QHD image (seed 11), 2560x1440 RGBA =
14,745,600 B raw, sha256 3e9bd5da0d45b20f... Code: qhd_test.py,
qhd_bench.py, fps_bench.py, frames_store.py, imp_driver.py,
driver_demo.py (workspace + Chrimbie/ImplicationSpace).
Original run logs live on Christopher's PC in exp_space_run
(fps_log.txt, frames_log.txt, driver_log2.txt, driver_stats.json).

## Single image, four treatments (VM, 2 vCPU, median of 3)

| method | size (B) | write | read |
|---|---|---|---|
| plain file | 14,745,600 | 2,726 MB/s | 1,077 MB/s |
| EVB2 store | 15,729,354 | 235 MB/s | 3,201 MB/s |
| IMPZ1 | 14,775,178 (1.002) | 0.7 MB/s | 92.3 MB/s |
| zip deflate | 441,092 (0.030) | 152 MB/s | 564.5 MB/s |

Random row window out of the store: 0.09 ms/row. Full
read-back sha match; all sampled windows pixel-exact.

## Steady-state frames per second (fps_bench.py)

VM (2 vCPU, 40 frames):
- ingest bulk 19.4 img/s; per-frame-save 10.4 img/s
- playback (hash-verified) 50.1 img/s, 40/40
- random access 291.8 img/s (store fits in page cache)
- derived playback (never-written) 4.5 img/s

Christopher's PC (i9-12900K, 60 frames):
- ingest bulk 12.7 img/s; per-frame-save 11.5 img/s
- playback (hash-verified) 27.5 img/s, 60/60
- random access 34.9 img/s
- derived playback 1.8 img/s
- PC slower than the VM here; Defender realtime is ON and
  the workload churns a ~2 GB file (consistent, not A/B-proven)
- store on disk 1,889,084,858 B for 120 frames

## The stored movie (frames_store.py, PC)

120 distinct animated frames stored persistently in
frames_qhd.evb2 (1,887,472,898 B) with a per-frame SHA-256
manifest. Ingest 5.6 img/s (includes CPU rendering of each
frame); playback 32.1 img/s, hash-verified 120/120. Proof
sheet frames_proof_sheet.png assembled from read-back bytes.

## The graphics driver (imp_driver.py, PC)

Headless (SDL dummy), 240 frames each:
- 2560x1440: 12.4 fps, dirty 0.47, store file 24.1 MB
- 1280x720: 43.1 fps, dirty 0.636, store file 7.3 MB

Live window (Christopher ran RUN_DRIVER.bat himself):
- 2560x1440: **16.0 fps**, dirty 0.443, 906.3 MB committed =
  906.3 MB presented, store still 24.1 MB
- 1280x720: **50.7 fps**, dirty 0.649, 1,039.4 MB through a
  7.3 MB store

Design note (hard lesson, same day): the first driver
version wrote row segments through the store's append-only
write and ballooned to 12.83 GB in ~150 s; fixed with
BlockStore.write_inplace + per-block batched commit/present
(50 in-place rewrites verified: file size unchanged).
