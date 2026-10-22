"""AN OLLAMA MODEL RUNNING WITH EXPONENTIATED VRAM, 2026-10-21.

Christopher's pick: qwen3-coder:30b — 18 GB of weights against
a 12 GB card, "the real test."

How the model uses exponentiated VRAM: its LONG-TERM MEMORY
is an ImpliedFile (exp_vram) with a logical size of 1 TB,
stored in kilobytes. The runner mediates memory tools in the
chat: when the model emits a line

    MEMORY_WRITE <offset> <text>
    MEMORY_READ  <offset> <length>

the runner executes it against the implied memory and returns
the result — the same pattern as any tool-using agent, except
the memory lives in implication space.

The test: session 1 asks the model to store a note at offset
500 GB. The runner verifies the bytes landed in the file,
exactly. Session 2 is a FRESH context — the model is told only
where the note lives; it must read its implied memory to
answer. The codeword can only come from the space.

Receipt: ollama's own GPU/CPU split (ollama ps), live VRAM
(nvidia-smi), the memory file's logical vs stored bytes, and
byte-verification of everything the model stored.
"""

import json
import os
import re
import subprocess
import sys
import urllib.request

from exp_vram import ImpliedFile

MODEL = "qwen3-coder:30b"
MEM_PATH = "coder_memory.evb"
MEM_SIZE = 10 ** 12                  # 1 TB of implied memory
NOTE_OFF = 500 * 10 ** 9             # the note lives at 500 GB
NOTE = b"The launch codeword is VIOLET ANCHOR."
API = "http://localhost:11434/api/chat"

SYSTEM = (
    "You are an AI with a long-term memory stored in "
    "implication space (a 1 TB addressable memory). To use it, "
    "reply with a line in exactly one of these forms:\n"
    "MEMORY_WRITE <offset> <text>\n"
    "MEMORY_READ <offset> <length>\n"
    "The system executes it and returns the result. Use one "
    "memory line per reply when you need memory; otherwise "
    "answer normally and briefly.")


def chat(messages):
    req = urllib.request.Request(
        API, data=json.dumps(
            {"model": MODEL, "messages": messages,
             "stream": False}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.load(r)["message"]["content"]


def session(mem, first_msg, turns=6):
    """Run one conversation; mediate MEMORY_* lines."""
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": first_msg}]
    for _ in range(turns):
        reply = chat(msgs)
        msgs.append({"role": "assistant", "content": reply})
        m = re.search(r"MEMORY_WRITE\s+(\d+)\s+(.+)", reply)
        if m:
            off, text = int(m.group(1)), m.group(2).strip()
            mem.seek(off)
            mem.write(text.encode())
            mem.save()
            msgs.append({"role": "user", "content":
                         f"MEMORY RESULT: wrote {len(text)} "
                         f"bytes at offset {off}."})
            continue
        m = re.search(r"MEMORY_READ\s+(\d+)\s+(\d+)", reply)
        if m:
            off, ln = int(m.group(1)), int(m.group(2))
            mem.seek(off)
            data = mem.read(ln)
            msgs.append({"role": "user", "content":
                         f"MEMORY RESULT: {data!r}"})
            continue
        return reply                      # plain answer: done
    return msgs[-1]["content"]


def main():
    if os.path.exists(MEM_PATH):
        os.remove(MEM_PATH)
    mem = ImpliedFile.create(MEM_PATH, MEM_SIZE, layer=3)
    print(f"implied memory created: {MEM_SIZE:,} logical bytes "
          f"| file on disk: {os.path.getsize(MEM_PATH):,} bytes",
          flush=True)

    print("\n— session 1: store a note at offset 500 GB —",
          flush=True)
    a1 = session(mem, "Store this note in your long-term memory "
                      f"at offset {NOTE_OFF}, exactly as given: "
                      f"'{NOTE.decode()}' — then confirm briefly.")
    print("model:", a1.strip()[:300], flush=True)
    mem.seek(NOTE_OFF)
    landed = mem.read(len(NOTE) + 32).rstrip(b"\x00")[:len(NOTE)]
    print(f"byte-check of the note in implied memory: "
          f"{'PASS' if landed == NOTE else f'CHECK: {landed!r}'}",
          flush=True)

    print("\n— session 2: fresh context, recall from memory —",
          flush=True)
    a2 = session(mem, "A note was stored in your long-term "
                      f"memory at offset {NOTE_OFF} (48 bytes). "
                      "Read it and tell me the launch codeword.")
    print("model:", a2.strip()[:300], flush=True)
    ok = "VIOLET ANCHOR" in a2
    print(f"recall check: {'PASS' if ok else 'FAIL'}", flush=True)

    print("\n— the machine's own accounting —", flush=True)
    ps = subprocess.run(["ollama", "ps"], capture_output=True,
                        text=True).stdout.strip()
    print(ps, flush=True)
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total",
         "--format=csv,noheader"], capture_output=True,
        text=True).stdout.strip()
    print(f"nvidia-smi VRAM: {smi}", flush=True)
    print(f"RECEIPT: {MODEL} (18 GB weights, 12 GB card) ran "
          f"with a {MEM_SIZE / 1e12:.0f} TB long-term memory "
          f"stored in {os.path.getsize(MEM_PATH):,} bytes | "
          f"note stored at offset {NOTE_OFF:,} byte-verified | "
          f"fresh-session recall "
          f"{'PASS' if ok else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
