"""cloud_server.py - the Implication Cloud interface.

Christopher's direction (2026-10-22): "let's turn it into a
cloud" - and "it needs to be able to save folders as well."
A storage cloud whose product is the SPACE: objects (single
files, or whole folders with their paths preserved) are
ingested into one EXP-VRAM BlockStore inside a 2^60-byte
envelope; the original bytes are kept nowhere else. Every
download is served out of the space; every object carries its
SHA-256 from ingest, and /prove re-reads the object from the
store and re-hashes it - the containment test the 70B passed,
as a button. Folders come back out as a .zip streamed from
the store.

Zero dependencies beyond numpy (the store). HTTP via stdlib.
v1 has NO auth: run it on a machine/network you trust; a key
gate belongs in front before any public exposure.
"""
import hashlib
import json
import os
import sys
import threading
import time
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from exp_blockstore import BlockStore

HERE = Path(__file__).resolve().parent
STORE_PATH = HERE / "cloud.evb2"
INDEX_PATH = HERE / "cloud_objects.json"
ENVELOPE = 1 << 60
FIRST_ADDR = 1 << 40          # first object lives 1 TiB deep
ALIGN = 1 << 20
CHUNK = 1 << 20

_lock = threading.Lock()
_store = None
_objects = {}                  # id -> {name, offset, length, sha256, ts}
_next = FIRST_ADDR


def _save_index():
    tmp = INDEX_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(
        {"next": _next, "objects": _objects}, indent=1))
    os.replace(tmp, INDEX_PATH)


def _open_store():
    global _store, _objects, _next
    if STORE_PATH.exists():
        _store = BlockStore.open(str(STORE_PATH))
    else:
        _store = BlockStore.create(str(STORE_PATH), ENVELOPE)
    if INDEX_PATH.exists():
        d = json.loads(INDEX_PATH.read_text())
        _objects = d["objects"]
        _next = d["next"]


def _alloc(n):
    global _next
    off = _next
    _next += (n + ALIGN - 1) // ALIGN * ALIGN
    return off


def space_stats():
    folders = {}
    for o in _objects.values():
        if "/" in o["name"]:
            root = o["name"].split("/", 1)[0]
            folders[root] = folders.get(root, 0) + 1
    return {
        "envelope_bytes": ENVELOPE,
        "stored_bytes": _store.stored_bytes(),
        "blocks": _store.blocks_stored(),
        "objects": len(_objects),
        "folders": folders,
        "content_bytes": sum(o["length"]
                             for o in _objects.values()),
    }


def sanitize_name(raw):
    """Keep a safe relative path: forward slashes, no '..',
    no leading slash/drive, no empty result."""
    name = unquote(raw).replace("\\", "/")
    parts = [p for p in name.split("/")
             if p not in ("", ".", "..")]
    name = "/".join(parts).strip()
    if name.startswith("/") or ":" in name.split("/")[0]:
        name = name.lstrip("/").split("/", 1)[-1] \
            if "/" in name else "unnamed"
    return name or "unnamed"


PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Implication Cloud - EXP-VRAM storage</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0b0e14; color:#dbe4f3;
    font:15px/1.55 ui-monospace, SFMono-Regular, Menlo, monospace; }
  main { max-width: 920px; margin: 0 auto; padding: 28px 18px 60px; }
  h1 { font-size: 26px; margin: 0 0 4px; letter-spacing: .5px; }
  h1 .thin { color:#5f7bb0; font-weight: 400; }
  .sub { color:#8fa3c8; margin-bottom: 22px; }
  .cards { display:grid; grid-template-columns:repeat(auto-fit,
    minmax(160px,1fr)); gap:10px; margin-bottom: 22px; }
  .card { background:#121826; border:1px solid #22304d;
    border-radius:10px; padding:12px 14px; }
  .card .k { color:#7d92b8; font-size:12px; text-transform:
    uppercase; letter-spacing: 1px; }
  .card .v { font-size:19px; margin-top:3px; color:#fff; }
  .panel { background:#101623; border:1px solid #22304d;
    border-radius:10px; padding:16px; margin-bottom: 20px; }
  button { background:#2b5cff; color:#fff; border:0;
    border-radius:7px; padding:8px 14px; font:inherit;
    cursor:pointer; }
  button.ghost { background:#1a2438; color:#b9c9e8;
    border:1px solid #2c3d5f; padding:4px 10px; }
  button:hover { filter:brightness(1.15); }
  table { width:100%; border-collapse: collapse; }
  td, th { padding:7px 6px; border-bottom:1px solid #1d2942;
    text-align:left; vertical-align:top; }
  th { color:#7d92b8; font-weight:400; font-size:12px;
    text-transform:uppercase; letter-spacing:1px; }
  .hash { color:#7ee2a8; word-break: break-all; }
  .addr { color:#e2c07e; }
  .ok { color:#7ee2a8; } .bad { color:#ff8a8a; }
  .muted { color:#7186ab; }
  input[type=file] { font:inherit; color:#b9c9e8; }
  #drop { border:1px dashed #33477a; border-radius:10px;
    padding:18px; text-align:center; color:#8fa3c8; }
  #drop.over { border-color:#2b5cff; color:#fff; }
  .folder { color:#9fd0ff; }
  .bar { background:#0b0e14; border:1px solid #22304d;
    border-radius:6px; height:14px; overflow:hidden;
    margin:4px 0 10px; }
  .bar > div { background:#2b5cff; height:100%; width:0%;
    transition: width .15s linear; }
  .bar.all > div { background:#2fbf71; }
  .barlbl { color:#8fa3c8; font-size:13px; display:flex;
    justify-content:space-between; }
</style></head><body><main>
<h1>IMPLICATION CLOUD <span class="thin">/ EXP-VRAM</span></h1>
<div class="sub">Storage with infinite space. Files and whole
folders live inside an implication space - the envelope is an
exbibyte; only the bytes you store are real. Everything is
served back out of the space and can be re-proven against
its hash.</div>
<div class="cards" id="cards"></div>
<div class="panel">
  <div id="drop">Drop files or a folder here, or
    <input type="file" id="file">
    <input type="file" id="dir" webkitdirectory>
    <button id="up">Put it in the space</button></div>
  <div id="upmsg" class="muted" style="margin-top:10px"></div>
  <div id="prog" style="display:none; margin-top:10px">
    <div class="barlbl"><span id="curfile">file</span>
      <span id="curpct">0%</span></div>
    <div class="bar"><div id="curbar"></div></div>
    <div class="barlbl"><span id="allfiles">overall</span>
      <span id="allpct">0%</span></div>
    <div class="bar all"><div id="allbar"></div></div>
  </div>
  <div id="folders" style="margin-top:10px"></div>
</div>
<div class="panel">
  <div style="display:flex; justify-content:space-between;
    align-items:center; margin-bottom:8px">
    <span class="muted">OBJECTS IN THE SPACE</span>
    <button class="ghost" id="delall" style="display:none;
      color:#ff9a9a; border-color:#5f2c2c">Delete all</button>
  </div>
  <table><thead><tr><th>Object</th><th>Size</th>
  <th>Address in the space</th><th>SHA-256</th><th></th>
  </tr></thead><tbody id="rows"></tbody></table>
  <div id="empty" class="muted">The space is empty. For now.</div>
</div>
<div class="muted">A Christopher Ladnier construction -
implication space / EXP-VRAM. The same store held a 70B
model and generated from inside it.</div>
<script>
const fmt = n => n >= 2**60 ? (n/2**60).toFixed(2)+' EiB'
  : n >= 2**50 ? (n/2**50).toFixed(2)+' PiB'
  : n >= 2**40 ? (n/2**40).toFixed(2)+' TiB'
  : n >= 2**30 ? (n/2**30).toFixed(2)+' GiB'
  : n >= 2**20 ? (n/2**20).toFixed(2)+' MiB'
  : n >= 2**10 ? (n/2**10).toFixed(1)+' KiB' : n+' B';
async function load() {
  const s = await (await fetch('/api/space')).json();
  const nf = Object.keys(s.folders || {}).length;
  document.getElementById('cards').innerHTML = [
    ['Envelope', fmt(s.envelope_bytes)],
    ['On disk', fmt(s.stored_bytes)],
    ['Objects', s.objects],
    ['Folders', nf],
    ['Content', fmt(s.content_bytes)],
  ].map(([k,v]) => `<div class="card"><div class="k">${k}
    </div><div class="v">${v}</div></div>`).join('');
  document.getElementById('folders').innerHTML =
    Object.entries(s.folders || {}).map(([root, n]) =>
      `<div><span class="folder">&#128193; ${root}/</span>
       <span class="muted">${n} files</span>
       <a href="/folder-zip?root=${encodeURIComponent(root)}">
       <button class="ghost">Download .zip</button></a></div>`
    ).join('');
  const os = await (await fetch('/api/objects')).json();
  document.getElementById('empty').style.display =
    os.length ? 'none' : 'block';
  document.getElementById('rows').innerHTML = os.map(o => `
    <tr><td>${o.name}</td><td>${fmt(o.length)}</td>
    <td class="addr">${o.offset.toLocaleString()}</td>
    <td class="hash">${o.sha256.slice(0,16)}&hellip;</td>
    <td style="white-space:nowrap">
      <a href="/objects/${o.id}"><button class="ghost">
      Get</button></a>
      <button class="ghost" onclick="prove('${o.id}', this)">
      Prove</button>
      <button class="ghost delbtn" data-id="${o.id}"
      data-name="${o.name}" style="color:#ff9a9a;
      border-color:#5f2c2c">Delete</button>
      </td></tr>`).join('');
  const da = document.getElementById('delall');
  da.style.display = os.length ? 'inline-block' : 'none';
  da.textContent = `Delete all (${os.length})`;
  document.querySelectorAll('.delbtn').forEach(b =>
    b.onclick = async () => {
      if (!confirm(`Delete ${b.dataset.name} from the space?`))
        return;
      await fetch('/objects/' + b.dataset.id,
        { method: 'DELETE' });
      load();
    });
}
document.getElementById('delall').onclick = async () => {
  const os = await (await fetch('/api/objects')).json();
  if (!os.length) return;
  if (!confirm(`Delete ALL ${os.length} objects from the ` +
    `space? The store resets to empty and the disk space is ` +
    `given back. This cannot be undone.`)) return;
  await fetch('/api/delete-all', { method: 'POST' });
  load();
};
async function prove(id, btn) {
  btn.textContent = 'Proving...';
  const r = await (await fetch(`/objects/${id}/prove`)).json();
  btn.outerHTML = r.match
    ? '<span class="ok">hash matches - contained</span>'
    : '<span class="bad">MISMATCH</span>';
}
function sendOne(file, path, onprog) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', '/objects?name=' +
      encodeURIComponent(path));
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) onprog(e.loaded, e.total); };
    xhr.onload = () => {
      if (xhr.status === 200) {
        try { resolve(JSON.parse(xhr.responseText)); }
        catch (err) { reject(err); }
      } else reject(new Error(path + ': ' + xhr.status));
    };
    xhr.onerror = () => reject(new Error(path + ': network'));
    xhr.send(file);
  });
}
async function sendAll(items) {
  const msg = document.getElementById('upmsg');
  const prog = document.getElementById('prog');
  const totalBytes = items.reduce((a, [f]) => a + f.size, 0);
  let done = 0, bytesDone = 0;
  prog.style.display = 'block';
  const setBar = (id, pct) => {
    document.getElementById(id).style.width =
      Math.min(100, pct).toFixed(1) + '%'; };
  for (const [file, path] of items) {
    document.getElementById('curfile').textContent =
      `Ingesting ${done + 1}/${items.length}: ${path}`;
    document.getElementById('allfiles').textContent =
      `Overall - ${done}/${items.length} files, ` +
      `${fmt(bytesDone)} of ${fmt(totalBytes)}`;
    try {
      await sendOne(file, path, (loaded, total) => {
        const cp = total ? loaded / total * 100 : 100;
        setBar('curbar', cp);
        document.getElementById('curpct').textContent =
          Math.floor(cp) + '%';
        const ap = totalBytes
          ? (bytesDone + loaded) / totalBytes * 100 : 100;
        setBar('allbar', ap);
        document.getElementById('allpct').textContent =
          Math.floor(ap) + '%';
      });
    } catch (e) {
      msg.textContent = 'Failed on ' + e.message;
      prog.style.display = 'none';
      return;
    }
    bytesDone += file.size;
    done++;
    setBar('curbar', 0);
    document.getElementById('curpct').textContent = '0%';
  }
  setBar('allbar', 100);
  document.getElementById('allpct').textContent = '100%';
  msg.textContent = `Done - ${done} file(s) now live in the space.`;
  setTimeout(() => { prog.style.display = 'none'; }, 2500);
  load();
}
async function walkEntry(entry, base, out) {
  if (entry.isFile) {
    const f = await new Promise((res, rej) =>
      entry.file(res, rej));
    out.push([f, base + entry.name]);
  } else if (entry.isDirectory) {
    const rd = entry.createReader();
    let batch;
    do {
      batch = await new Promise((res, rej) =>
        rd.readEntries(res, rej));
      for (const e of batch)
        await walkEntry(e, base + entry.name + '/', out);
    } while (batch.length);
  }
}
document.getElementById('up').onclick = async () => {
  const items = [];
  const f1 = document.getElementById('file').files[0];
  if (f1) items.push([f1, f1.name]);
  for (const f of document.getElementById('dir').files)
    items.push([f, f.webkitRelativePath || f.name]);
  if (items.length) sendAll(items);
};
const drop = document.getElementById('drop');
drop.ondragover = e => { e.preventDefault();
  drop.classList.add('over'); };
drop.ondragleave = () => drop.classList.remove('over');
drop.ondrop = async e => {
  e.preventDefault(); drop.classList.remove('over');
  const items = [];
  const entries = [...e.dataTransfer.items]
    .map(i => i.webkitGetAsEntry && i.webkitGetAsEntry())
    .filter(Boolean);
  if (entries.length) {
    for (const en of entries) await walkEntry(en, '', items);
  } else {
    for (const f of e.dataTransfer.files) items.push([f, f.name]);
  }
  if (items.length) sendAll(items);
};
load();
</script></main></body></html>"""


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s %s\n" % (self.address_string(),
                                      fmt % args))

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type",
                             "text/html; charset=utf-8")
            self.send_header("Content-Length", len(body))
            self.end_headers()
            self.wfile.write(body)
        elif u.path == "/api/space":
            with _lock:
                self._json(space_stats())
        elif u.path == "/api/objects":
            with _lock:
                self._json([{"id": k, **v} for k, v in
                            sorted(_objects.items(),
                                   key=lambda kv: kv[1]["ts"])])
        elif u.path == "/folder-zip":
            root = parse_qs(u.query).get("root", [""])[0]
            self._folder_zip(unquote(root))
        elif u.path.startswith("/objects/"):
            rest = u.path[len("/objects/"):]
            if rest.endswith("/prove"):
                self._prove(rest[:-len("/prove")])
            else:
                self._download(rest)
        else:
            self._json({"error": "not found"}, 404)

    def _download(self, oid):
        with _lock:
            o = _objects.get(oid)
        if o is None:
            return self._json({"error": "no such object"}, 404)
        self.send_response(200)
        self.send_header("Content-Type",
                         "application/octet-stream")
        self.send_header(
            "Content-Disposition",
            f'attachment; filename='
            f'"{os.path.basename(o["name"])}"')
        self.send_header("Content-Length", o["length"])
        self.end_headers()
        pos = 0
        with _lock:
            while pos < o["length"]:
                n = min(CHUNK, o["length"] - pos)
                self.wfile.write(
                    _store.read(o["offset"] + pos, n))
                pos += n

    def _folder_zip(self, root):
        with _lock:
            members = [(oid, o) for oid, o in _objects.items()
                       if o["name"] == root or
                       o["name"].startswith(root + "/")]
        if not members:
            return self._json({"error": "no such folder"}, 404)
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header(
            "Content-Disposition",
            f'attachment; filename="{root}.zip"')
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        zf = zipfile.ZipFile(self.wfile, "w",
                             zipfile.ZIP_STORED)
        with _lock:
            for _oid, o in sorted(members,
                                  key=lambda kv: kv[1]["name"]):
                info = zipfile.ZipInfo(
                    o["name"], date_time=(2026, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o644 << 16
                with zf.open(info, "w") as fh:
                    pos = 0
                    while pos < o["length"]:
                        n = min(CHUNK, o["length"] - pos)
                        fh.write(_store.read(
                            o["offset"] + pos, n))
                        pos += n
        zf.close()

    def _prove(self, oid):
        with _lock:
            o = _objects.get(oid)
            if o is None:
                return self._json({"error": "no such object"}, 404)
            h = hashlib.sha256()
            pos = 0
            while pos < o["length"]:
                n = min(CHUNK, o["length"] - pos)
                h.update(_store.read(o["offset"] + pos, n))
                pos += n
            digest = h.hexdigest()
        self._json({"id": oid, "sha256": digest,
                    "expected": o["sha256"],
                    "match": digest == o["sha256"]})

    def do_DELETE(self):
        u = urlparse(self.path)
        if not u.path.startswith("/objects/"):
            return self._json({"error": "not found"}, 404)
        oid = u.path[len("/objects/"):]
        with _lock:
            o = _objects.pop(oid, None)
            if o is None:
                return self._json({"error": "no such object"}, 404)
            _save_index()
        self._json({"deleted": oid, "name": o["name"]})

    def _delete_all(self):
        global _store, _objects, _next
        with _lock:
            n = len(_objects)
            _store.close()
            os.remove(STORE_PATH)
            _store = BlockStore.create(str(STORE_PATH), ENVELOPE)
            _objects = {}
            _next = FIRST_ADDR
            _save_index()
        self._json({"deleted_all": n})

    def do_POST(self):
        u = urlparse(self.path)
        if u.path == "/api/delete-all":
            return self._delete_all()
        if u.path != "/objects":
            return self._json({"error": "not found"}, 404)
        name = sanitize_name(
            parse_qs(u.query).get("name", ["unnamed"])[0])
        length = int(self.headers.get("Content-Length", 0))
        with _lock:
            off = _alloc(length)
            h = hashlib.sha256()
            pos = 0
            while pos < length:
                n = min(CHUNK, length - pos)
                data = self.rfile.read(n)
                if not data:
                    break
                h.update(data)
                _store.write(off + pos, data)
                pos += len(data)
            # A body that stops early (tab closed, connection
            # cut) records only the bytes that actually
            # arrived - never the declared length. Anything
            # else would leave a phantom object whose tail
            # silently reads back as derivation.
            length = pos
            oid = (h.hexdigest()[:16] + "-" +
                   str(int(time.time() * 1000)))
            _objects[oid] = {"name": name, "offset": off,
                             "length": length,
                             "sha256": h.hexdigest(),
                             "ts": time.time()}
            _store.save()
            _save_index()
        self._json({"id": oid, "name": name, "offset": off,
                    "length": length,
                    "sha256": h.hexdigest()})


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8899
    _open_store()
    print(f"Implication Cloud on :{port} | envelope "
          f"{ENVELOPE:,} B | stored "
          f"{_store.stored_bytes():,} B | "
          f"{len(_objects)} objects", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), Handler) \
        .serve_forever()


if __name__ == "__main__":
    main()
