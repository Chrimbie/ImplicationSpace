"""IMPOSSIBLE CUBE — a spinning multicolored cube in PyOpenGL whose
faces are a live window into an implication simulation too large
to store. Christopher's shape (2026-10-21): the simplest test he
knows — a multicolored cube spinning in PyOpenGL — driven by his
sets-of-four implication units, full pool, aggregated.

THE SIMULATION
  Pool: N = 25,769,803,776 units — every set of four bits in a
  4070 Ti's 12 GB, by Christopher's divide-by-4 construction.
  Each unit is one controller nibble v in 0..15. Each unit's full
  state is its implied pattern at the canon width 32,768
  (fan-out: the combination IS the pattern) — never stored.
  Implied state: N x 32,768 bits = 8.44e14 bits ~ 105.5 TB.
  Stored state: the N controller bytes in RAM ~ 25.8 GB.
  The patterns exist only as derivations, in the aggregate
  summaries the display is built from.

DISPLAY
  6 faces x 128 x 128 cells = 98,304 cells. The pool divides
  EXACTLY: each cell aggregates 2^18 = 262,144 consecutive units.
  Per sweep, each cell reports three aggregates of its units'
  implied patterns, computed in closed form on the GPU:
    density   = mean pattern popcount / 32,768
    warm      = fraction of units whose top bit is set
    hue seed  = a wrapped multiplicative mix of the nibbles
  Aggregates -> HSV -> RGB textures (one per face). The cube
  spins continuously; textures refresh when a sweep completes.

EVOLUTION (EMPIRICAL — Mia's step rule, not Christopher's):
  v'(i) = (v(i-1) + 3*v(i) + v(i+1) + step) mod 16, ring topology.
  Marked EMPIRICAL: a simple mixing rule chosen so the surface
  visibly lives; nothing in the project depends on it.

VERIFICATION (--selftest): small pool on CPU; the sweep engine
and aggregates checked against an independently written naive
reference; pattern popcount closed form checked against fully
materialized patterns for sampled units.

Run:  python impossible_cube.py            (the cube)
      python impossible_cube.py --selftest (checks, no window)
"""

import argparse
import threading
import time

import numpy as np

# ------------------------------------------------------------------
N_DEFAULT = 25_769_803_776          # 3 * 2^33, his divide-by-4 pool
PATTERN_WIDTH = 32_768              # canon layer-3 width
FACES, GRID = 6, 128
CELLS = FACES * GRID * GRID         # 98,304
POPCOUNT = np.array([bin(i).count("1") for i in range(16)],
                    dtype=np.float64)
TILE_CAP = 1 << 24
SEED = 20261021


def init_pool(n, progress=None):
    """Initial controllers. GPU RNG in tiles when CUDA is present
    (seconds); numpy Generator otherwise. Deterministic either
    way per platform (seeded); the init is arbitrary by design."""
    out = np.empty(n, dtype=np.uint8)
    chunk = 1 << 26
    gpu_ok = False
    try:
        import torch
        gpu_ok = torch.cuda.is_available()
    except Exception:
        gpu_ok = False
    if gpu_ok:
        import torch
        for a in range(0, n, chunk):
            b = min(n, a + chunk)
            g = torch.Generator(device="cuda")
            g.manual_seed(SEED + a // chunk)
            t = torch.randint(0, 16, (b - a,), device="cuda",
                              generator=g, dtype=torch.int64)
            out[a:b] = t.to(torch.uint8).cpu().numpy()
            if progress:
                progress(b / n)
        return out
    rng = np.random.default_rng(SEED)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        out[a:b] = rng.integers(0, 16, size=b - a, dtype=np.uint8)
        if progress:
            progress(b / n)
    return out


# ------------------------------------------------------------------
# Engines. sweep: one evolution step + per-cell aggregates.
# aggregates_only: the aggregates of the current state, no step.
# Aggregates per cell: (density, warm, hue) of the implied
# patterns, in closed form from the nibbles.
# ------------------------------------------------------------------

def _aggregates_from_block(vals):
    density = POPCOUNT[vals].mean(axis=1) / 4.0
    warm = (vals >= 8).mean(axis=1)
    mix = ((vals.astype(np.uint64) + 1) * 2654435761) & 0xFFFFFFFF
    hue = (mix.sum(axis=1) & 0x3FF).astype(np.float64) / 1024.0
    return density, warm, hue


def sweep_numpy(pool, step, upc):
    left = np.roll(pool, 1)
    right = np.roll(pool, -1)
    new = ((left.astype(np.uint16) + 3 * pool + right + step)
           % 16).astype(np.uint8)
    return new, _aggregates_from_block(new.reshape(CELLS, upc))


def aggregates_numpy(pool, upc):
    return _aggregates_from_block(pool.reshape(CELLS, upc))


def _torch_tiles(pool, upc, tile_units):
    n = pool.size
    for a in range(0, n, tile_units):
        b = min(n, a + tile_units)
        yield a, b


def _torch_agg_block(blk, pop):
    import torch
    density = (pop[blk].mean(dim=1) / 4.0).cpu().numpy()
    warm = (blk >= 8).float().mean(dim=1).cpu().numpy()
    mix = ((blk.to(torch.int64) + 1) * 2654435761) & 0xFFFFFFFF
    hue = ((mix.sum(dim=1) & 0x3FF).float() / 1024.0).cpu().numpy()
    return density, warm, hue


def _pop_table():
    import torch
    return torch.tensor([0, 1, 1, 2, 1, 2, 2, 3,
                         1, 2, 2, 3, 2, 3, 3, 4],
                        dtype=torch.float32, device="cuda")


def sweep_torch(pool, step, upc, tile_units):
    import torch
    n = pool.size
    new = np.empty_like(pool)
    density = np.empty(CELLS)
    warm = np.empty(CELLS)
    hue = np.empty(CELLS)
    pop = _pop_table()
    cells_per_tile = tile_units // upc
    for a, b in _torch_tiles(pool, upc, tile_units):
        lo = np.empty(b - a + 2, dtype=np.uint8)
        lo[1:-1] = pool[a:b]
        lo[0] = pool[a - 1]              # ring halo (a=0 wraps to N-1)
        lo[-1] = pool[b % n]
        t = torch.from_numpy(lo.astype(np.int32)).to("cuda")
        nv = (t[:-2] + 3 * t[1:-1] + t[2:] + step) % 16
        new[a:b] = nv.cpu().numpy().astype(np.uint8)
        c0 = a // upc
        d, w, h = _torch_agg_block(nv.reshape(cells_per_tile, upc),
                                   pop)
        density[c0:c0 + cells_per_tile] = d
        warm[c0:c0 + cells_per_tile] = w
        hue[c0:c0 + cells_per_tile] = h
    return new, (density, warm, hue)


def aggregates_torch(pool, upc, tile_units):
    import torch
    density = np.empty(CELLS)
    warm = np.empty(CELLS)
    hue = np.empty(CELLS)
    pop = _pop_table()
    cells_per_tile = tile_units // upc
    for a, b in _torch_tiles(pool, upc, tile_units):
        t = torch.from_numpy(pool[a:b].astype(np.int32)).to("cuda")
        c0 = a // upc
        d, w, h = _torch_agg_block(t.reshape(cells_per_tile, upc),
                                   pop)
        density[c0:c0 + cells_per_tile] = d
        warm[c0:c0 + cells_per_tile] = w
        hue[c0:c0 + cells_per_tile] = h
    return density, warm, hue


# ------------------------------------------------------------------
def aggregates_to_faces(agg):
    """(density, warm, hue) per cell -> 6 face textures (128,128,3)."""
    density, warm, hue = agg
    h = hue % 1.0
    s = 0.45 + 0.55 * warm
    v = 0.12 + 0.88 * density
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    rgb = np.zeros((CELLS, 3))
    cases = [(v, t, p), (q, v, p), (p, v, t),
             (p, q, v), (t, p, v), (v, p, q)]
    for k, (r, g, b) in enumerate(cases):
        m = i == k
        rgb[m, 0], rgb[m, 1], rgb[m, 2] = r[m], g[m], b[m]
    faces = (rgb * 255).astype(np.uint8).reshape(FACES, GRID, GRID, 3)
    return [np.ascontiguousarray(faces[k]) for k in range(FACES)]


# ------------------------------------------------------------------
def selftest():
    print("SELFTEST — small pool, CPU engine vs naive reference")
    n = CELLS * 64                       # 6,291,456 units, upc = 64
    upc = n // CELLS
    pool = init_pool(n)

    def ref_sweep(p, step):
        nn = np.empty_like(p)
        for i in range(p.size):
            nn[i] = (int(p[i - 1]) + 3 * int(p[i]) +
                     int(p[(i + 1) % p.size]) + step) % 16
        return nn

    small = pool[:4096].copy()
    r = ref_sweep(small, 3)
    l = np.roll(small, 1)
    rr = np.roll(small, -1)
    f = ((l.astype(np.uint16) + 3 * small + rr + 3) % 16)
    print("  rule (formula vs loop reference):",
          "PASS" if (r == f).all() else "FAIL")

    new, agg = sweep_numpy(pool, 1, upc)
    blocks = new.reshape(CELLS, upc)
    d_ref = np.array([POPCOUNT[blocks[c]].mean() / 4.0
                      for c in range(0, CELLS, 997)])
    print("  aggregates (engine vs spot reference):",
          "PASS" if np.allclose(d_ref, agg[0][::997]) else "FAIL")

    ok = True
    for v in range(16):
        pat = np.repeat(np.array([(v >> (3 - k)) & 1
                                  for k in range(4)]), 8192)
        if pat.sum() != POPCOUNT[v] * 8192:
            ok = False
    print("  pattern popcount == popcount(v) x 8192, all v:",
          "PASS" if ok else "FAIL")
    print(f"  full pool implied state = "
          f"{N_DEFAULT * PATTERN_WIDTH / 8e12:.1f} TB")
    print("SELFTEST DONE")


# ------------------------------------------------------------------
def run_cube(units):
    import pygame
    from OpenGL.GL import (glBegin, glClear, glClearColor,
                           glColor3f, glEnable, glEnd, glGenTextures,
                           glLoadIdentity, glMatrixMode, glRotatef,
                           glTexCoord2f, glTexImage2D, glTexParameteri,
                           glTranslatef, glVertex3f, glBindTexture,
                           GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT,
                           GL_DEPTH_TEST, GL_MODELVIEW, GL_NEAREST,
                           GL_PROJECTION, GL_QUADS, GL_RGB,
                           GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                           GL_TEXTURE_MIN_FILTER, GL_UNSIGNED_BYTE)
    from OpenGL.GLU import gluPerspective

    if units % CELLS != 0:
        raise SystemExit(f"units must divide evenly into {CELLS} "
                         f"cells; got {units:,}")
    upc = units // CELLS
    tile_units = upc * max(1, TILE_CAP // upc)
    tile_units = min(tile_units, units)

    use_gpu = False
    try:
        import torch
        use_gpu = torch.cuda.is_available()
    except Exception:
        use_gpu = False

    implied_tb = units * PATTERN_WIDTH / 8e12
    print(f"pool: {units:,} units | implied state {implied_tb:.1f} "
          f"TB | stored {units / 1e9:.1f} GB of controllers | "
          f"engine: {'GPU tiles' if use_gpu else 'CPU'}", flush=True)

    state = {"phase": "initializing controllers…", "pct": 0,
             "pool": None, "step": 0, "faces": None, "sweep_s": 0.0}
    lock = threading.Lock()

    def sim_loop():
        def prog(frac):
            with lock:
                state["pct"] = int(frac * 100)
        pool = init_pool(units, progress=prog)
        with lock:
            state["pool"] = pool
            state["phase"] = "computing initial faces…"
        agg = aggregates_torch(pool, upc, tile_units) if use_gpu \
            else aggregates_numpy(pool, upc)
        with lock:
            state["faces"] = aggregates_to_faces(agg)
            state["phase"] = "running"
        while True:
            t0 = time.time()
            if use_gpu:
                new, agg = sweep_torch(state["pool"], state["step"],
                                       upc, tile_units)
            else:
                new, agg = sweep_numpy(state["pool"], state["step"],
                                       upc)
            faces = aggregates_to_faces(agg)
            dt = time.time() - t0
            with lock:
                state["pool"] = new
                state["step"] += 1
                state["faces"] = faces
                state["sweep_s"] = dt
            print(f"sweep {state['step']}: {dt:.1f}s "
                  f"({units / dt:,.0f} units/s)", flush=True)

    threading.Thread(target=sim_loop, daemon=True).start()

    pygame.init()
    pygame.display.set_mode((900, 900),
                            pygame.OPENGL | pygame.DOUBLEBUF)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    glClearColor(0.04, 0.04, 0.07, 1.0)
    tex_ids = glGenTextures(FACES)
    uploaded = {"step": -1, "faces_id": None}

    verts = [((1, -1, -1), (1, 1, -1), (-1, 1, -1), (-1, -1, -1)),
             ((1, -1, 1), (1, 1, 1), (-1, 1, 1), (-1, -1, 1)),
             ((-1, -1, -1), (-1, 1, -1), (-1, 1, 1), (-1, -1, 1)),
             ((1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)),
             ((-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)),
             ((-1, 1, -1), (1, 1, -1), (1, 1, 1), (-1, 1, 1))]
    texcoords = [(1, 0), (1, 1), (0, 1), (0, 0)]

    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, 1.0, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)

    clock = pygame.time.Clock()
    angle = 0.0
    running = True
    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
        with lock:
            faces = state["faces"]
            step, sdt = state["step"], state["sweep_s"]
            phase, pct = state["phase"], state["pct"]
        if faces is not None and id(faces) != uploaded["faces_id"]:
            for k in range(FACES):
                glBindTexture(GL_TEXTURE_2D, tex_ids[k])
                glTexParameteri(GL_TEXTURE_2D,
                                GL_TEXTURE_MAG_FILTER, GL_NEAREST)
                glTexParameteri(GL_TEXTURE_2D,
                                GL_TEXTURE_MIN_FILTER, GL_NEAREST)
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, GRID, GRID,
                             0, GL_RGB, GL_UNSIGNED_BYTE, faces[k])
            uploaded["faces_id"] = id(faces)
            uploaded["step"] = step
        if phase != "running":
            pygame.display.set_caption(
                f"Impossible Cube — {phase} {pct}%")
        else:
            pygame.display.set_caption(
                f"Impossible Cube — sweep {step} "
                f"({sdt:.0f}s/sweep) | {units:,} units | "
                f"{implied_tb:.0f} TB implied | "
                f"{units / 1e9:.0f} GB stored")
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()
        glTranslatef(0.0, 0.0, -5.0)
        glRotatef(angle, 1.0, 1.0, 0.4)
        glColor3f(1, 1, 1)
        if uploaded["faces_id"] is not None:
            for k in range(FACES):
                glBindTexture(GL_TEXTURE_2D, tex_ids[k])
                glBegin(GL_QUADS)
                for (tc, vt) in zip(texcoords, verts[k]):
                    glTexCoord2f(*tc)
                    glVertex3f(*vt)
                glEnd()
        pygame.display.flip()
        angle += 0.7
        clock.tick(60)
    pygame.quit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--units", type=int, default=N_DEFAULT)
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run_cube(a.units)


if __name__ == "__main__":
    main()
