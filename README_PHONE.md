# Implied Framebuffer — phone run (CPU only)

Graphics memory inside implication space, running entirely on
the phone's CPU. No GPU is used or needed.

You get: a 32,768 x 32,768 RGBA canvas = 4 GiB of logical
graphics memory, backed by a store file of ~1.3 GB (only what
you draw is stored; the rest is derived). The run draws a
composition, presents three windows, and checks each one
pixel-exact against an independent render — then reopens the
store and checks again.

## What you need

- Termux (from F-Droid or the Play Store)
- About 2 GB free storage

## Steps (in Termux)

1. Install Python + numpy (prebuilt, no compiling):

       pkg update && pkg install python-numpy

2. Allow Termux to see your Downloads folder:

       termux-setup-storage

3. Put these two files in Downloads (they must sit together):

       implied_framebuffer.py
       exp_blockstore.py

4. Run it:

       cd ~/storage/downloads
       python implied_framebuffer.py --phone

   Expect ~1-3 minutes on a phone CPU (it took 22 s on a
   desktop-class VM core). It prints PASS lines as each
   window verifies.

5. See the presented frame:

       framebuffer_phone_present.png

   (in the same folder — open it from your file manager)

## The GPU partition test (gpu_partition.py)

Partitions the phone's CPU: a quarter of its cores (2 of the
Pixel's 8 - the performance pair) act as a GPU, rendering the
same composition tile-by-tile in parallel into the implied
framebuffer. The serial single-core draw runs first as the
baseline; the partition must then produce a pixel-identical
picture, faster. Workers report the cores they actually ran
on, so the partition is verified, not assumed.

    python gpu_partition.py --phone

Expect a few minutes. It ends with "GPU PARTITION TEST:
ALL PASS" and a one-line JSON summary - send that output back
to Mia. Phone-lean: the serial store is deleted mid-run, so
peak storage stays around one store (~1.3 GB). Worth being
plugged in for this one - several minutes of full CPU is hard
on a tired battery.

## Notes

- `framebuffer_phone.evb2` is the store — the actual graphics
  memory on disk. Delete it whenever; the canvas is gone with
  it, like pulling a GPU.
- The static-looking background in the presented frame is
  implication itself: pixels nobody drew, derived on read.
- Everything is plain Python + numpy. The same two files run
  the 16 GiB canvas on a PC with no flag at all.
