"""imp_driver.py - a graphics driver whose framebuffer lives
in implication space.

Christopher's direction (2026-10-22): "let's turn this into
a graphics driver." Shape (his tap): a user-space driver -
programs written against it open a Display, draw, and present.
Not a Windows display driver; Windows itself cannot see it
(same line as EXP-VRAM: usable by programs written against
it).

Architecture:
  canvas   - an RGBA numpy surface the program draws on
  commit() - writes only DIRTY TILES of the canvas into the
             EVB2 BlockStore, at the back buffer's address in
             a 2^40 envelope
  present()- reads the just-committed tiles BACK OUT of the
             store and blits them to the real window - the
             window is fed from the space, never from RAM
  buffers  - two frame slots in the store, swapped per frame

So a frame's cost is what changed, not what exists: the
present path moves dirty tiles only (TILE=256 px, EMPIRICAL
tuning choice). Counters keep the receipts: bytes committed,
bytes presented, dirty fraction per frame.
"""
import os
import sys

import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from exp_blockstore import BlockStore

TILE = 256                     # px, EMPIRICAL
ENVELOPE = 1 << 40


class Display:
    def __init__(self, width, height, store_path, base_addr=0):
        self.w, self.h = int(width), int(height)
        self.frame_bytes = self.w * self.h * 4
        self.stride = (self.frame_bytes + (1 << 20) - 1) \
            // (1 << 20) * (1 << 20)
        self.base = base_addr
        if os.path.exists(store_path):
            self.store = BlockStore.open(store_path)
        else:
            self.store = BlockStore.create(store_path, ENVELOPE)
        self.canvas = np.zeros((self.h, self.w, 4),
                               dtype=np.uint8)
        self.canvas[:, :, 3] = 255
        self.tiles_x = (self.w + TILE - 1) // TILE
        self.tiles_y = (self.h + TILE - 1) // TILE
        self.dirty = set()
        self.front = 0           # slot currently on screen
        self.bytes_committed = 0
        self.bytes_presented = 0
        self.frames = 0
        self.all_dirty()

    # ---------------- dirty tracking ----------------------
    def _tile_range(self, x0, y0, x1, y1):
        tx0, ty0 = max(x0, 0) // TILE, max(y0, 0) // TILE
        tx1 = min(self.w, x1) // TILE
        ty1 = min(self.h, y1) // TILE
        for ty in range(ty0, ty1 + 1):
            for tx in range(tx0, tx1 + 1):
                self.dirty.add((tx, ty))

    def all_dirty(self):
        self.dirty = {(tx, ty) for tx in range(self.tiles_x)
                      for ty in range(self.tiles_y)}

    def dirty_fraction(self):
        return len(self.dirty) / (self.tiles_x * self.tiles_y)

    def _tile_bounds(self, tx, ty):
        x0, y0 = tx * TILE, ty * TILE
        return x0, y0, min(x0 + TILE, self.w), \
            min(y0 + TILE, self.h)

    # ---------------- drawing -----------------------------
    def fill(self, color):
        self.canvas[:, :] = color
        self.all_dirty()

    def blit_array(self, img, x=0, y=0):
        """img: (h, w, 4) uint8 RGBA."""
        hh, ww = img.shape[:2]
        x1, y1 = min(self.w, x + ww), min(self.h, y + hh)
        if x1 <= x or y1 <= y:
            return
        self.canvas[y:y1, x:x1] = img[:y1 - y, :x1 - x]
        self._tile_range(x, y, x1 - 1, y1 - 1)

    def restore_tiles(self, bg, tiles):
        for tx, ty in tiles:
            x0, y0, x1, y1 = self._tile_bounds(tx, ty)
            self.canvas[y0:y1, x0:x1] = bg[y0:y1, x0:x1]
            self.dirty.add((tx, ty))

    def rect(self, x, y, w, h, color):
        x1, y1 = min(self.w, x + w), min(self.h, y + h)
        x, y = max(x, 0), max(y, 0)
        if x1 <= x or y1 <= y:
            return
        self.canvas[y:y1, x:x1] = color
        self._tile_range(x, y, x1 - 1, y1 - 1)

    def circle(self, cx, cy, r, color):
        x0, x1 = max(0, int(cx - r)), min(self.w, int(cx + r) + 1)
        y0, y1 = max(0, int(cy - r)), min(self.h, int(cy + r) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        yy, xx = np.mgrid[y0:y1, x0:x1]
        m = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        region = self.canvas[y0:y1, x0:x1]
        region[m] = color
        self.canvas[y0:y1, x0:x1] = region
        self._tile_range(x0, y0, x1 - 1, y1 - 1)

    # ---------------- the space ---------------------------
    def _slot_addr(self, slot):
        return self.base + slot * self.stride

    def _segments(self, tiles):
        """Frame-byte ranges (start, end) a tile set covers."""
        segs = []
        for tx, ty in tiles:
            x0, y0, x1, y1 = self._tile_bounds(tx, ty)
            for y in range(y0, y1):
                s = (y * self.w + x0) * 4
                segs.append((s, s + (x1 - x0) * 4))
        return segs

    def _blocks(self, tiles, addr):
        """store block index -> [(frame_start, frame_end)]."""
        bs = self.store.block_size
        by_block = {}
        for s, e in self._segments(tiles):
            a, b = addr + s, addr + e
            bi = a // bs
            while bi <= (b - 1) // bs:
                by_block.setdefault(bi, []).append((s, e))
                bi += 1
        return by_block

    def commit(self):
        """Write dirty tiles into the back slot, batched per
        store block and in place: each touched block is spliced
        once and overwritten, so the store file stays the size
        of the frames it holds no matter how hot they run."""
        back = 1 - self.front
        addr = self._slot_addr(back)
        bs = self.store.block_size
        moved = 0
        for bi, segs in self._blocks(self.dirty, addr).items():
            bstart = bi * bs
            blen = min(bs, ENVELOPE - bstart)
            blk = bytearray(self.store.read(bstart, blen))
            fb0 = bstart - addr          # frame byte at blk start
            for s, e in segs:
                y = s // (self.w * 4)
                x0 = (s % (self.w * 4)) // 4
                n = (e - s) // 4
                blk[s - fb0:e - fb0] = \
                    self.canvas[y, x0:x0 + n].tobytes()
                moved += e - s
            self.store.write_inplace(bstart, bytes(blk))
        self.bytes_committed += moved
        self._pending = (back, set(self.dirty))
        self.dirty = set()
        return moved

    def present(self, surf_array):
        """Feed the window from the store: read the committed
        tiles out of the space (one read per touched block)
        and into the pygame surface array ((w, h, 3), RGB)."""
        back, tiles = self._pending
        addr = self._slot_addr(back)
        bs = self.store.block_size
        blocks = {}
        for bi in self._blocks(tiles, addr):
            bstart = bi * bs
            blocks[bi] = (bstart - addr,
                          self.store.read(bstart,
                                          min(bs, ENVELOPE
                                              - bstart)))
        moved = 0
        for tx, ty in tiles:
            x0, y0, x1, y1 = self._tile_bounds(tx, ty)
            rows = []
            for y in range(y0, y1):
                s = (y * self.w + x0) * 4
                e = s + (x1 - x0) * 4
                bi = (addr + s) // bs
                fb0, blk = blocks[bi]
                rows.append(blk[s - fb0:e - fb0])
            tile = np.frombuffer(b"".join(rows),
                                 dtype=np.uint8).reshape(
                                     y1 - y0, x1 - x0, 4)
            surf_array[x0:x1, y0:y1] = np.ascontiguousarray(
                tile[:, :, :3].transpose(1, 0, 2))
            moved += (x1 - x0) * (y1 - y0) * 4
        self.bytes_presented += moved
        self.front = back
        self.frames += 1
        return moved

    def save(self):
        self.store.save()

    def close(self):
        self.store.close()
