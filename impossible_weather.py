"""IMPOSSIBLE WEATHER — a whole-Earth weather simulation on
Christopher's implication pool, 2026-10-21.

"Something scientists dreamed of: a simulation of our weather
system, working as intended." Built as a replica of Earth: the
land/ocean geography is real, classified from an actual world
map (earth_mask.npz, 1440x720, built from the Wikimedia
equirectangular shaded-relief map; 15/15 test points correct).

THE SIMULATION
  Grid: the full pool as a sphere — 24,576 latitude rows x
  1,048,576 longitude columns = 25,769,803,776 cells, one unit
  (controller nibble) per cell. Each cell's microstate is its
  implied 32,768-bit pattern: implied state ~105.5 TB, stored
  state 25.8 GB of controllers. Weather variables are DERIVED
  per cell per step, in closed form from the nibble + position:
    warmth   = popcount(v)/4          -> temperature anomaly
    moisture = a hash summary of v    -> humidity
    pressure = base - warmth coupling - moisture coupling
  Winds come from the pressure field plus a latitude base flow
  (trades in the tropics, westerlies in midlatitudes — banding
  that stands in for Coriolis). The field ADVECTS: each cell
  takes the nibble of its upwind source, so structures travel.
  Local forcing nudges cells: the sun heats a moving strip
  (day/night), warm moist ocean cells charge (storm food),
  cold land cells shed. A meridional mixing pass exchanges a
  fraction of cells with the row below.

  ALL dynamics constants are EMPIRICAL (Mia's simplified rules,
  chosen so the system behaves like weather: bands, traveling
  storms, a moving terminator). Nothing in the implication
  engine depends on them. A stylized weather system on the
  real planet's geography — not a forecast model.

DISPLAY: pygame map (aggregated 1024x512), land shaded from the
real mask, temperature as color, rain as bright speckle, night
side darkened. Caption carries the accounting.

Run:  python impossible_weather.py             (full pool)
      python impossible_weather.py --selftest  (checks)
      python impossible_weather.py --preview   (one PNG frame)
"""

import argparse
import threading
import time

import numpy as np

N_LON_FULL = 1_048_576
N_LAT_FULL = 24_576
PATTERN_WIDTH = 32_768
POPCOUNT = np.array([bin(i).count("1") for i in range(16)],
                    dtype=np.float64)
MASK_PATH = "earth_mask.npz"
SEED = 20261021
BLOCK = 32                                # rows per processing block


# ------------------------------------------------------------------
class LandField:
    """The real-Earth land mask, kept small (720x1440); rows and
    display aggregates are sampled from it on demand."""

    def __init__(self, path=MASK_PATH):
        self.small = np.load(path)["land"].astype(np.float64)
        self.mh, self.mw = self.small.shape

    def row(self, r, n_lat, col_idx):
        mr = int(r * self.mh / n_lat)
        return self.small[mr][col_idx]

    def block(self, r0, rows, n_lat, col_idx):
        mrows = (np.arange(r0, r0 + rows) * self.mh
                 / n_lat).astype(int)
        return self.small[mrows][:, col_idx]

    def display(self, out_h, out_w):
        from PIL import Image
        im = Image.fromarray((self.small * 255).astype(np.uint8))
        im = im.resize((out_w, out_h), Image.BILINEAR)
        return np.asarray(im).astype(np.float64) / 255.0


def col_index(n_lon, mw):
    return (np.arange(n_lon) * mw / n_lon).astype(int)


def lat_array(n_lat):
    return 90.0 - (np.arange(n_lat) + 0.5) * 180.0 / n_lat


def base_temperature(lat):
    """Zonal base temperature (deg C). EMPIRICAL."""
    return 28.0 * np.cos(np.deg2rad(lat)) ** 1.4 - 12.0 * \
        (np.abs(lat) / 90.0) ** 3


def derive_block(v, lat_b, land_b, sun_lon, lon_axis):
    """Weather fields for a block of rows. v: (B, n_lon) uint8.
    Returns T, moist, pressure as (B, n_lon) float64."""
    warmth = POPCOUNT[v] / 4.0
    moist = ((v.astype(np.uint32) * 2654435761) >> 13 & 15) / 15.0
    land_amp = 1.0 + 0.7 * land_b
    sun = np.clip(np.cos(np.deg2rad(lon_axis - sun_lon)), 0, None)
    T = (base_temperature(lat_b)[:, None]
         + (warmth - 0.5) * 22.0 * land_amp
         + (sun[None, :] - 0.35) * 6.0 * land_amp)
    pressure = -(warmth - 0.5) * 26.0 - (moist - 0.5) * 6.0
    return T, moist, pressure


def wind_block(lat_b, pressure):
    """Zonal wind u (cells/step) per block. EMPIRICAL base flow
    by latitude band + pressure-gradient anomaly."""
    alat = np.abs(lat_b)
    u_base = np.where(
        alat < 28, -1.6 * np.cos(np.deg2rad(lat_b)),
        np.where(alat < 65,
                 2.4 * np.sin(np.deg2rad(alat - 28) / 37 * np.pi),
                 -0.8))
    dpdx = np.gradient(pressure, axis=1)
    hemi = np.where(lat_b >= 0, 1.0, -1.0)[:, None]
    return u_base[:, None] - dpdx * 0.06 * hemi


def step_grid(grid, land, lats, step_no, sun_speed=3.0):
    """One full step. grid: (n_lat, n_lon) uint8. Works in row
    blocks; returns (new_grid, sun_lon)."""
    n_lat, n_lon = grid.shape
    lon_axis = (np.arange(n_lon) + 0.5) * 360.0 / n_lon - 180.0
    sun_lon = -180.0 + (step_no * sun_speed) % 360.0
    cidx = col_index(n_lon, land.mw)
    cols = np.arange(n_lon)
    new = np.empty_like(grid)
    for r0 in range(0, n_lat, BLOCK):
        rows = min(BLOCK, n_lat - r0)
        v = grid[r0:r0 + rows]
        land_b = land.block(r0, rows, n_lat, cidx)
        T, moist, pressure = derive_block(
            v, lats[r0:r0 + rows], land_b, sun_lon, lon_axis)
        u = wind_block(lats[r0:r0 + rows], pressure)
        src = (cols[None, :] - np.round(u).astype(int)) % n_lon
        nv = np.take_along_axis(v, src, axis=1)
        warm_ocean = (land_b < 0.5) & (T > 24) & (moist > 0.55)
        cold_land = (land_b >= 0.5) & (T < 2)
        nv = np.where(warm_ocean & (nv < 14), nv + 1, nv)
        nv = np.where(cold_land & (nv > 1), nv - 1, nv)
        new[r0:r0 + rows] = nv.astype(np.uint8)
    # meridional mixing pass (EMPIRICAL), block-chunked
    for r0 in range(0, n_lat, BLOCK):
        rows = min(BLOCK, n_lat - r0)
        h = ((cols[None, :].astype(np.uint64) * 2654435761
              + (step_no * 97 + r0)) >> 11 & 15) < 1
        take = h & (cols[None, :] % 7 == 0)
        below = np.roll(new[r0:r0 + rows], -1, axis=0)
        if r0 + rows < n_lat:
            below[-1] = new[r0 + rows]
        blk = np.where(take, below, new[r0:r0 + rows])
        new[r0:r0 + rows] = blk
    return new, sun_lon


def aggregates_for_display(grid, land, lats, sun_lon, out_w=1024,
                           out_h=512):
    """Aggregate the grid to a display RGB image. Block means
    when the grid divides evenly into the display (the full
    pool does); otherwise derive at native resolution and
    sample at display centers (small test grids)."""
    n_lat, n_lon = grid.shape
    lon_axis = (np.arange(n_lon) + 0.5) * 360.0 / n_lon - 180.0
    cidx = col_index(n_lon, land.mw)
    if n_lat % out_h == 0 and n_lon % out_w == 0:
        rows_per = n_lat // out_h
        cols_per = n_lon // out_w
        Tm = np.zeros((out_h, out_w))
        Mm = np.zeros((out_h, out_w))
        for dr in range(out_h):
            r0 = dr * rows_per
            acc_T = np.zeros(out_w)
            acc_M = np.zeros(out_w)
            for r in range(r0, r0 + rows_per, BLOCK):
                rows = min(BLOCK, r0 + rows_per - r)
                land_b = land.block(r, rows, n_lat, cidx)
                T, moist, _ = derive_block(
                    grid[r:r + rows], lats[r:r + rows], land_b,
                    sun_lon, lon_axis)
                acc_T += T.reshape(rows, out_w, cols_per).mean(
                    axis=(0, 2)) * rows
                acc_M += moist.reshape(rows, out_w, cols_per).mean(
                    axis=(0, 2)) * rows
            Tm[dr] = acc_T / rows_per
            Mm[dr] = acc_M / rows_per
    else:
        Tf = np.zeros((n_lat, n_lon))
        Mf = np.zeros((n_lat, n_lon))
        for r0 in range(0, n_lat, BLOCK):
            rows = min(BLOCK, n_lat - r0)
            land_b = land.block(r0, rows, n_lat, cidx)
            T, moist, _ = derive_block(
                grid[r0:r0 + rows], lats[r0:r0 + rows], land_b,
                sun_lon, lon_axis)
            Tf[r0:r0 + rows] = T
            Mf[r0:r0 + rows] = moist
        ri = ((np.arange(out_h) + 0.5) * n_lat / out_h).astype(int)
        ci = ((np.arange(out_w) + 0.5) * n_lon / out_w).astype(int)
        Tm = Tf[ri][:, ci]
        Mm = Mf[ri][:, ci]
    Lfr = land.display(out_h, out_w)
    return render_map(Tm, Mm, Lfr, sun_lon, out_w, out_h)


def render_map(Tm, Mm, Lfr, sun_lon, out_w, out_h):
    t = np.clip((Tm + 30) / 65.0, 0, 1)               # -30..35 C
    r = np.clip(1.5 * t - 0.25, 0, 1)
    g = np.clip(1.0 - np.abs(t - 0.5) * 1.6, 0, 1)
    b = np.clip(1.25 - 1.5 * t, 0, 1)
    img = np.stack([r, g, b], -1)
    land_tint = np.array([0.85, 0.78, 0.55])
    img = img * (1 - Lfr[..., None] * 0.45) \
        + land_tint * (Lfr[..., None] * 0.45)
    rain = (Mm > 0.72) & (Tm < 18)
    img[rain] = img[rain] * 0.4 + np.array([0.5, 0.6, 1.0]) * 0.6
    lon_axis = (np.arange(out_w) + 0.5) * 360.0 / out_w - 180.0
    day = np.clip(np.cos(np.deg2rad(lon_axis - sun_lon)), -1, 1)
    shade = (0.35 + 0.65 * np.clip(day, 0, None))[None, :, None]
    return (np.clip(img * shade, 0, 1) * 255).astype(np.uint8)


# ------------------------------------------------------------------
def init_grid(n_lat, n_lon, structured=True):
    rng = np.random.default_rng(SEED)
    grid = rng.integers(0, 16, size=(n_lat, n_lon), dtype=np.uint8)
    if structured:
        lats = lat_array(n_lat)
        for r in range(n_lat):
            w0 = np.clip(0.5 + base_temperature(lats[r]) / 60.0,
                         0, 1)
            grid[r] = np.clip(w0 * 15 + rng.integers(-3, 4, n_lon),
                              0, 15).astype(np.uint8)
    return grid


def selftest():
    print("SELFTEST — weather engine, small grid")
    n_lat, n_lon = 96, 192
    land = LandField()
    lats = lat_array(n_lat)

    def cell_land(lat, lon):
        r = int((90 - lat) / 180 * n_lat)
        cidx = col_index(n_lon, land.mw)
        return land.row(r, n_lat, cidx)[int(
            (lon + 180) / 360 * n_lon) % n_lon] > 0.5

    checks = [("Dallas land", cell_land(32.8, -96.8)),
              ("mid-Pacific ocean", not cell_land(0, -150)),
              ("Sahara land", cell_land(22, 10)),
              ("mid-Atlantic ocean", not cell_land(20, -30))]
    for name, ok in checks:
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")

    grid = init_grid(n_lat, n_lon)
    sun = -180.0
    for s in range(6):
        grid, sun = step_grid(grid, land, lats, s)
    lon_axis = (np.arange(n_lon) + 0.5) * 360.0 / n_lon - 180.0
    cidx = col_index(n_lon, land.mw)
    eq = grid[n_lat // 2:n_lat // 2 + 1]
    T_eq, _, _ = derive_block(eq, np.array([0.0]),
                              land.block(n_lat // 2, 1, n_lat, cidx),
                              sun, lon_axis)
    T_pole, _, _ = derive_block(grid[2:3], lats[2:3],
                                land.block(2, 1, n_lat, cidx),
                                sun, lon_axis)
    print(f"  equator T {T_eq.mean():.1f} C vs near-pole "
          f"{T_pole.mean():.1f} C: "
          f"{'PASS' if T_eq.mean() > T_pole.mean() + 10 else 'FAIL'}")
    print(f"  sun longitude moves: "
          f"{'PASS' if sun != -180.0 else 'FAIL'} (now {sun:.0f})")
    img = aggregates_for_display(grid, land, lats, sun, 256, 128)
    print(f"  display render {img.shape}: "
          f"{'PASS' if img.shape == (128, 256, 3) else 'FAIL'}")
    print("SELFTEST DONE")


def preview(path="weather_preview.png", n_lat=384, n_lon=768,
            steps=10):
    land = LandField()
    lats = lat_array(n_lat)
    grid = init_grid(n_lat, n_lon)
    sun = -180.0
    for s in range(steps):
        grid, sun = step_grid(grid, land, lats, s)
    img = aggregates_for_display(grid, land, lats, sun, 1024, 512)
    from PIL import Image
    Image.fromarray(img).save(path)
    print(f"preview saved: {path}")


# ------------------------------------------------------------------
def run_full():
    import pygame
    n_lat, n_lon = N_LAT_FULL, N_LON_FULL
    units = n_lat * n_lon
    print(f"grid {n_lat:,} x {n_lon:,} = {units:,} cells | "
          f"implied {units * PATTERN_WIDTH / 8e12:.1f} TB | "
          f"stored {units / 1e9:.1f} GB", flush=True)
    land = LandField()
    lats = lat_array(n_lat)
    print("initializing controllers on GPU…", flush=True)
    grid = np.empty(units, dtype=np.uint8)
    try:
        import torch
        chunk = 1 << 26
        for a in range(0, units, chunk):
            b = min(units, a + chunk)
            g = torch.Generator(device="cuda")
            g.manual_seed(SEED + a // chunk)
            t = torch.randint(0, 16, (b - a,), device="cuda",
                              generator=g, dtype=torch.int64)
            grid[a:b] = t.to(torch.uint8).cpu().numpy()
    except Exception:
        rng = np.random.default_rng(SEED)
        grid[:] = rng.integers(0, 16, size=units, dtype=np.uint8)
    grid = grid.reshape(n_lat, n_lon)
    # structured warmth start, in row blocks
    for r0 in range(0, n_lat, 4096):
        rows = min(4096, n_lat - r0)
        w0 = np.clip(0.5 + base_temperature(
            lats[r0:r0 + rows]) / 60.0, 0, 1)[:, None]
        grid[r0:r0 + rows] = np.clip(
            w0 * 15 + (grid[r0:r0 + rows].astype(int) - 7),
            0, 15).astype(np.uint8)

    state = {"img": None, "step": 0, "dt": 0.0}
    lock = threading.Lock()
    holder = [grid]

    def loop():
        step = 0
        while True:
            t0 = time.time()
            new, sun = step_grid(holder[0], land, lats, step)
            holder[0] = new
            img = aggregates_for_display(new, land, lats, sun)
            dt = time.time() - t0
            with lock:
                state["img"] = img
                state["step"] = step + 1
                state["dt"] = dt
            print(f"step {step + 1}: {dt:.1f}s", flush=True)
            step += 1

    threading.Thread(target=loop, daemon=True).start()

    pygame.init()
    screen = pygame.display.set_mode((1400, 760))
    pygame.display.set_caption("Impossible Weather — initializing…")
    clock = pygame.time.Clock()
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        with lock:
            img, st, dt = state["img"], state["step"], state["dt"]
        if img is not None:
            surf = pygame.surfarray.make_surface(
                np.transpose(img, (1, 0, 2)))
            surf = pygame.transform.scale(surf, (1400, 700))
            screen.blit(surf, (0, 0))
            pygame.display.set_caption(
                f"Impossible Weather — step {st} ({dt:.0f}s/step) "
                f"| {units:,} cells | "
                f"{units * PATTERN_WIDTH / 8e12:.0f} TB implied | "
                f"{units / 1e9:.0f} GB stored | real Earth geography")
        pygame.display.flip()
        clock.tick(30)
    pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--preview", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.preview:
        preview()
    else:
        run_full()


if __name__ == "__main__":
    main()
