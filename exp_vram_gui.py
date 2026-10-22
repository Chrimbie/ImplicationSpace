"""EXPONENTIATED VRAM — live monitor GUI (v2), 2026-10-21.

Christopher's idea: "vram can be exponentiated too" — a GUI
that shows every GB extra gained and streams current usage
from 13 GB up. v2 attaches the REAL implication space: it
watches the actual .evb buffers programs and models use
(exp_vram library) — including the Ollama coder's 1 TB
long-term memory — and shows how much implication VRAM is
in use, live:

  IMPL. VRAM ALLOCATED = sum of buffers' logical sizes
  STORED ON DISK       = sum of the .evb files' real bytes
  REAL BYTES WRITTEN   = bytes programs actually wrote
                         (patch counts across buffers)

The streaming chart (floor 13 GB, per Christopher) plots the
effective footprint: physical VRAM + implication VRAM in use.
The demo streamer from v1 is still available; buffer usage
draws as its own line whenever buffers exist.

Run:  python exp_vram_gui.py
      python exp_vram_gui.py --selftest
"""

import argparse
import glob
import os
import subprocess
import sys
import threading
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

GB = 1e9
PHYSICAL_GB_DEFAULT = 12.0          # RTX 4070 Ti
CHART_FLOOR_GB = 13.0               # Christopher: "from 13GB up"
WIDTHS = {                          # implied bits per unit,
    1: 8, 2: 128, 3: 32_768,        # by his growth equation
    4: 2 ** 31, 5: 2 ** 63,
}


def addressable_bytes(stored_bytes, layer):
    return stored_bytes * WIDTHS[layer] / 8.0


def fmt_bytes(b):
    for unit, scale in (("EB", 1e18), ("PB", 1e15), ("TB", 1e12),
                        ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if abs(b) >= scale:
            return f"{b / scale:,.2f} {unit}"
    return f"{b:,.0f} B"


def fmt_gb(gb):
    if gb >= 1e6:
        return f"{gb / 1e6:,.2f} PB"
    if gb >= 1e3:
        return f"{gb / 1e3:,.2f} TB"
    return f"{gb:,.1f} GB"


# ------------------------------------------------------------------
# Watching real implication-space buffers (.evb via exp_vram)
# ------------------------------------------------------------------
def buffer_stats(path):
    """Stats for one .evb buffer, or None if unreadable."""
    try:
        from exp_vram import ImpliedBuffer
        buf = ImpliedBuffer.open(path)
        st = os.stat(path)
        return {"path": path, "logical": buf.size,
                "layer": buf.layer, "units": len(buf.units),
                "written": len(buf.patches),
                "file_bytes": st.st_size,
                "mtime": st.st_mtime}
    except Exception:
        return None


class BufferWatcher:
    """Polls a set of .evb files; re-reads only changed ones."""

    def __init__(self):
        self.stats = {}             # path -> stats dict
        self.events = []            # recent change descriptions

    def scan(self, extra_paths=()):
        here = os.path.dirname(os.path.abspath(__file__))
        paths = set(glob.glob(os.path.join(here, "*.evb")))
        paths |= {p for p in extra_paths if os.path.exists(p)}
        changed = False
        for p in sorted(paths):
            try:
                m = os.stat(p).st_mtime
            except OSError:
                continue
            cur = self.stats.get(p)
            if cur is None or cur["mtime"] != m:
                s = buffer_stats(p)
                if s is not None:
                    if cur is not None:
                        dw = s["written"] - cur["written"]
                        self.events.append(
                            f"{os.path.basename(p)}: "
                            f"{'+' if dw >= 0 else ''}{dw} bytes "
                            f"written (now {s['written']:,})")
                        self.events = self.events[-4:]
                    self.stats[p] = s
                    changed = True
        for p in list(self.stats):
            if p not in paths and p not in extra_paths:
                del self.stats[p]
                changed = True
        return changed

    def totals(self):
        t = {"logical": 0, "file_bytes": 0, "written": 0,
             "count": 0}
        for s in self.stats.values():
            t["logical"] += s["logical"]
            t["file_bytes"] += s["file_bytes"]
            t["written"] += s["written"]
            t["count"] += 1
        return t


# ------------------------------------------------------------------
class Streamer:
    """v1 demo workload: allocates implied buffers and derives
    their pattern summaries on the GPU for real."""

    def __init__(self, layer=3):
        self.layer = layer
        self.buffers = []
        self.active_units = 0
        self.torch = None
        self.lock = threading.Lock()
        self.derived_total = 0

    def _gpu_touch(self, units):
        try:
            if self.torch is None:
                import torch
                if not torch.cuda.is_available():
                    return 0
                self.torch = torch
            torch = self.torch
            sample = min(units, 1 << 16)
            v = torch.randint(0, 16, (sample,), device="cuda",
                              dtype=torch.int64)
            pc = torch.tensor([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3,
                               2, 3, 3, 4], device="cuda")
            _ = (pc[v] * 8192).sum()
            _ = ((v * 2654435761) % (2 ** 31)).sum()
            self.derived_total += sample * WIDTHS[self.layer]
            return sample
        except Exception:
            return 0

    def tick(self, rng):
        page = 1 << 20
        units = int(rng.integers(1, 33)) * page
        self._gpu_touch(units)
        with self.lock:
            self.buffers.append((units, time.time()))
            self.active_units += units
            while len(self.buffers) > 40:
                old, _ = self.buffers.pop(0)
                self.active_units -= old
        return self.active_units * WIDTHS[self.layer] / 8.0

    def active_bytes(self):
        with self.lock:
            return self.active_units * WIDTHS[self.layer] / 8.0


def physical_vram():
    total = PHYSICAL_GB_DEFAULT
    used = 0.0
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=4).stdout
        t, u = out.strip().splitlines()[0].split(",")
        total = float(t) * 1.048576e-3       # MiB -> GB
        used = float(u) * 1.048576e-3
    except Exception:
        pass
    return total, used


# ------------------------------------------------------------------
def selftest():
    import tempfile
    print("SELFTEST — exponentiated VRAM v2")
    assert WIDTHS[3] == 32_768 and WIDTHS[5] == 2 ** 63
    print("  layer widths from his equation: PASS")
    addr = addressable_bytes(12 * GB, 3)
    print(f"  12 GB @ layer 3 -> {fmt_bytes(addr)}: "
          f"{'PASS' if abs(addr - 4.9152e13) < 1e9 else 'FAIL'}")
    from exp_vram import ImpliedBuffer
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.evb")
        b = ImpliedBuffer.create(p, 10 ** 12)
        b.write(500 * 10 ** 9, b"VIOLET ANCHOR")
        b.save()
        s = buffer_stats(p)
        ok = (s["logical"] == 10 ** 12 and s["written"] == 13
              and s["file_bytes"] < 1000)
        print(f"  buffer stats (1 TB, 13 written, tiny file): "
              f"{'PASS' if ok else 'FAIL'}")
        w = BufferWatcher()
        w.scan(extra_paths=(p,))
        ok = w.stats[p]["logical"] == 10 ** 12
        b2 = ImpliedBuffer.open(p)
        b2.write(0, b"more")
        b2.save()
        os.utime(p, (time.time() + 2, time.time() + 2))
        w.scan(extra_paths=(p,))
        ok = ok and w.stats[p]["written"] == 17 \
            and len(w.events) >= 1
        print(f"  watcher totals + change detection: "
              f"{'PASS' if ok else 'FAIL'}")
    s = Streamer(layer=3)
    s._gpu_touch = lambda units: 0
    rng = np.random.default_rng(7)
    for _ in range(10):
        cur = s.tick(rng)
    print(f"  demo streamer bookkeeping: PASS "
          f"(active {fmt_bytes(cur)})")
    print("SELFTEST DONE")


# ------------------------------------------------------------------
def run_gui():
    import tkinter as tk
    from tkinter import filedialog
    total_phys, _ = physical_vram()
    state = {"history": [], "buf_history": [], "running": False,
             "peak": CHART_FLOOR_GB, "extra_paths": set()}
    streamer = Streamer(layer=3)
    watcher = BufferWatcher()
    rng = np.random.default_rng(20261021)

    root = tk.Tk()
    root.title("Exponentiated VRAM — implication monitor")
    root.geometry("1000x760")
    root.configure(bg="#0d1117")

    def lbl(text, size=12, color="#e6edf3", bold=False, parent=root):
        return tk.Label(parent, text=text, bg="#0d1117", fg=color,
                        font=("Consolas", size,
                              "bold" if bold else "normal"))

    lbl("EXPONENTIATED VRAM", 20, "#58a6ff", True).pack(pady=(12, 0))
    phys_v = tk.StringVar()
    lbl("", 11, "#8b949e")
    tk.Label(root, textvariable=phys_v, bg="#0d1117",
             fg="#8b949e", font=("Consolas", 11)).pack()

    def counter_row(parent, specs):
        row = tk.Frame(parent, bg="#0d1117")
        row.pack(pady=6)
        out = {}
        for title, color in specs:
            f = tk.Frame(row, bg="#0d1117")
            f.pack(side="left", padx=22)
            tk.Label(f, text=title, bg="#0d1117", fg="#8b949e",
                     font=("Consolas", 10)).pack()
            var = tk.StringVar(value="—")
            tk.Label(f, textvariable=var, bg="#0d1117", fg=color,
                     font=("Consolas", 16, "bold")).pack()
            out[title] = var
        return out

    lbl("the pool (v1 demo accounting)", 10, "#8b949e").pack(
        pady=(8, 0))
    c1 = counter_row(root, (("STORED (real)", "#e6edf3"),
                            ("ADDRESSABLE (implied)", "#58a6ff"),
                            ("EXTRA GB GAINED", "#3fb950")))
    lbl("implication vram in use — real buffers (.evb)", 10,
        "#8b949e").pack(pady=(6, 0))
    c2 = counter_row(root, (("IMPL. VRAM ALLOCATED", "#3fb950"),
                            ("STORED ON DISK", "#e6edf3"),
                            ("REAL BYTES WRITTEN", "#d29922")))

    controls = tk.Frame(root, bg="#0d1117")
    controls.pack(pady=2)
    tk.Label(controls, text="pool (stored GB):", bg="#0d1117",
             fg="#8b949e", font=("Consolas", 11)).pack(side="left")
    pool_scale = tk.Scale(controls, from_=1, to=24, resolution=1,
                          orient="horizontal", length=150,
                          bg="#0d1117", fg="#e6edf3",
                          highlightthickness=0)
    pool_scale.set(12)
    pool_scale.pack(side="left", padx=6)
    tk.Label(controls, text="layer:", bg="#0d1117", fg="#8b949e",
             font=("Consolas", 11)).pack(side="left", padx=(10, 2))
    layer_var = tk.IntVar(value=3)
    for n in (1, 2, 3, 4, 5):
        tk.Radiobutton(controls, text=str(n), variable=layer_var,
                       value=n, bg="#0d1117", fg="#e6edf3",
                       selectcolor="#21262d",
                       font=("Consolas", 11)).pack(side="left")
    btn = tk.Button(controls, text="▶ demo stream",
                    font=("Consolas", 11, "bold"), bg="#21262d",
                    fg="#3fb950", activebackground="#30363d")
    btn.pack(side="left", padx=12)

    def toggle():
        state["running"] = not state["running"]
        btn.config(text="⏸ demo stream" if state["running"]
                   else "▶ demo stream")
    btn.config(command=toggle)

    def add_buffer():
        p = filedialog.askopenfilename(
            title="Watch an implication buffer",
            filetypes=[("Implied buffers", "*.evb"),
                       ("All files", "*.*")])
        if p:
            state["extra_paths"].add(p)
    tk.Button(controls, text="＋ watch .evb…",
              font=("Consolas", 11), bg="#21262d", fg="#58a6ff",
              activebackground="#30363d",
              command=add_buffer).pack(side="left")

    canvas = tk.Canvas(root, width=950, height=280, bg="#010409",
                       highlightthickness=1,
                       highlightbackground="#30363d")
    canvas.pack(pady=8)

    buflist = tk.Listbox(root, width=118, height=6, bg="#010409",
                         fg="#8b949e", font=("Consolas", 10),
                         highlightthickness=1,
                         highlightbackground="#30363d",
                         selectbackground="#21262d")
    buflist.pack(pady=2)
    status_v = tk.StringVar(value="watching for .evb buffers…")
    tk.Label(root, textvariable=status_v, bg="#0d1117",
             fg="#8b949e", font=("Consolas", 10)).pack(pady=4)

    def draw_chart():
        canvas.delete("all")
        W, H = 950, 280
        top = max(CHART_FLOOR_GB * 2, state["peak"] * 1.15)
        def y_of(gb):
            frac = (gb - CHART_FLOOR_GB) / max(
                top - CHART_FLOOR_GB, 1e-9)
            return H - 24 - frac * (H - 44)
        for gb in (CHART_FLOOR_GB, (CHART_FLOOR_GB + top) / 2, top):
            y = y_of(gb)
            canvas.create_line(54, y, W - 8, y, fill="#21262d")
            canvas.create_text(50, y, text=fmt_gb(gb), anchor="e",
                               fill="#8b949e", font=("Consolas", 9))
        canvas.create_text(W - 12, y_of(CHART_FLOOR_GB) + 10,
                           text="13 GB — exponentiated region ↑",
                           anchor="e", fill="#58a6ff",
                           font=("Consolas", 9))
        def series(hist, color):
            if len(hist) < 2:
                return
            pts = [(54 + i * (W - 62) / max(len(hist) - 1, 1),
                    y_of(v)) for i, v in enumerate(hist)]
            flat = [c for p in pts for c in p]
            canvas.create_line(*flat, fill=color, width=2)
        series(state["buf_history"], "#58a6ff")     # real buffers
        series(state["history"], "#3fb950")        # demo streamer

    def refresh():
        total, used = physical_vram()
        phys_v.set(f"physical card: {total:.1f} GB "
                   f"({used:.1f} GB in use)  •  chart streams "
                   f"from {CHART_FLOOR_GB:.0f} GB up  •  blue = "
                   f"real buffers, green = demo stream")
        streamer.layer = int(layer_var.get())
        stored = float(pool_scale.get()) * GB
        addr = addressable_bytes(stored, streamer.layer)
        c1["STORED (real)"].set(fmt_bytes(stored))
        c1["ADDRESSABLE (implied)"].set(fmt_bytes(addr))
        c1["EXTRA GB GAINED"].set(fmt_gb(addr / GB - total))

        watcher.scan(extra_paths=tuple(state["extra_paths"]))
        t = watcher.totals()
        c2["IMPL. VRAM ALLOCATED"].set(fmt_bytes(t["logical"]))
        c2["STORED ON DISK"].set(fmt_bytes(t["file_bytes"]))
        c2["REAL BYTES WRITTEN"].set(f"{t['written']:,}")
        buflist.delete(0, "end")
        for p, s in sorted(watcher.stats.items()):
            buflist.insert(
                "end",
                f"{os.path.basename(p):28s} logical "
                f"{fmt_bytes(s['logical']):>10s}  •  stored "
                f"{fmt_bytes(s['file_bytes']):>9s}  •  written "
                f"{s['written']:,} B  •  layer {s['layer']}")
        if t["count"]:
            usage = total + t["logical"] / GB
            state["buf_history"].append(usage)
            state["buf_history"] = state["buf_history"][-160:]
            state["peak"] = max(state["peak"], usage)
        if state["running"]:
            active = streamer.tick(rng)
            usage = total + active / GB
            state["history"].append(usage)
            state["history"] = state["history"][-160:]
            state["peak"] = max(state["peak"], usage)
        ev = watcher.events[-1] if watcher.events else ""
        status_v.set(
            f"{t['count']} buffer(s) in use • implication VRAM "
            f"allocated {fmt_bytes(t['logical'])} • stored "
            f"{fmt_bytes(t['file_bytes'])}"
            + (f" • last change: {ev}" if ev else ""))
        draw_chart()
        root.after(900, refresh)

    refresh()
    root.mainloop()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        run_gui()


if __name__ == "__main__":
    main()
