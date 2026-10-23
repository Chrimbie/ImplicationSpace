"""driver_demo.py - live window demo for the implication
graphics driver (imp_driver.py).

The orbiting-balls scene, drawn through the driver: canvas ->
dirty tiles -> committed into the space -> presented back out
of the store into a real pygame window. Runs 2560x1440, then
1280x720, ~8 s each (ESC or close the window to stop early).
Stats land in driver_stats.json next to this file.

  python driver_demo.py              live window
  python driver_demo.py --headless   SDL dummy video (CI/selftest)
"""
import json
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HEADLESS = "--headless" in sys.argv
if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

import numpy as np
import pygame

from imp_driver import Display

SECONDS = 8.0
BALLS = [(0.20, 0.49, 0.111, (220, 80, 60, 255)),
         (0.51, 0.46, 0.153, (240, 200, 90, 255)),
         (0.80, 0.50, 0.097, (90, 200, 140, 255))]


def build_bg(w, h):
    rng = np.random.default_rng(11)
    img = np.zeros((h, w, 4), dtype=np.uint8)
    sky = int(h * 0.40)
    grad = np.linspace(30, 200, sky, dtype=np.uint8)
    img[:sky, :, 0] = grad[:, None] // 3
    img[:sky, :, 1] = grad[:, None] // 2
    img[:sky, :, 2] = grad[:, None]
    mid_end = int(h * 0.75)
    img[sky:mid_end, :] = (24, 24, 32, 255)
    tex_h = h - mid_end
    noise = rng.normal(128, 40, (tex_h // 8 + 2, w // 8 + 2, 3))
    big = np.kron(noise, np.ones((8, 8, 1)))[:tex_h, :w]
    img[mid_end:, :, :3] = np.clip(big, 0, 255).astype(np.uint8)
    img[:, :, 3] = 255
    return img


def run_res(w, h):
    store = os.path.join(HERE, f"impgfx_{w}x{h}.evb2")
    disp = Display(w, h, store)
    screen = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"ImpGfx {w}x{h}")
    surf = pygame.surfarray.array3d(screen)
    bg = build_bg(w, h)
    disp.blit_array(bg)
    disp.commit()
    disp.present(surf)
    pygame.surfarray.blit_array(screen, surf)
    pygame.display.flip()

    prev_tiles = set()
    frames = 0
    dirty_frac_sum = 0.0
    t0 = time.perf_counter()
    last_title = 0.0
    clock = pygame.time.Clock()
    running = True
    while running:
        now = time.perf_counter()
        if not HEADLESS and now - t0 > SECONDS:
            break
        if HEADLESS and frames >= 240:
            break
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT or (
                    ev.type == pygame.KEYDOWN
                    and ev.key == pygame.K_ESCAPE):
                running = False
        t = frames / 60.0
        ph = 2 * np.pi * t / 2.0
        if prev_tiles:
            disp.restore_tiles(bg, prev_tiles)
        for fx, fy, fr, col in BALLS:
            cx = (fx + 0.10 * np.sin(ph + fx * 9.0)) * w
            cy = (fy + 0.05 * np.sin(2 * ph + fx * 5.0)) * h
            disp.circle(cx, cy, fr * h, col)
        last_frac = disp.dirty_fraction()
        dirty_frac_sum += last_frac
        frame_tiles = set(disp.dirty)
        disp.commit()
        disp.present(surf)
        prev_tiles = frame_tiles
        pygame.surfarray.blit_array(screen, surf)
        pygame.display.flip()
        frames += 1
        if now - last_title > 0.5:
            fps_now = frames / max(now - t0, 1e-9)
            pygame.display.set_caption(
                f"ImpGfx {w}x{h} - {fps_now:.1f} fps - "
                f"dirty {last_frac:.0%}")
            last_title = now
        clock.tick(240)
    secs = time.perf_counter() - t0
    stats = {"res": f"{w}x{h}", "frames": frames,
             "seconds": round(secs, 2),
             "avg_fps": round(frames / secs, 1),
             "avg_dirty_fraction": round(
                 dirty_frac_sum / max(frames, 1), 3),
             "committed_MB": round(disp.bytes_committed / 1e6, 1),
             "presented_MB": round(disp.bytes_presented / 1e6, 1)}
    disp.save()
    disp.close()
    stats["store_MB"] = round(os.path.getsize(store) / 1e6, 1)
    print(json.dumps(stats), flush=True)
    return stats, running


def main():
    pygame.init()
    all_stats = []
    for w, h in [(2560, 1440), (1280, 720)]:
        stats, cont = run_res(w, h)
        all_stats.append(stats)
        if not cont:
            break
    pygame.quit()
    json.dump(all_stats, open(os.path.join(
        HERE, "driver_stats.json"), "w"), indent=1)
    print("DRIVER DEMO DONE", flush=True)


if __name__ == "__main__":
    main()
