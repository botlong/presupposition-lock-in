"""Record a content-addressed snapshot manifest for a model BEFORE running it.

Some caches (e.g. a local ModelScope download) expose no commit hash, so a pinned revision
is unavailable. A revision string alone is also mutable upstream; the sha256 manifest of the
files that determine behavior (config, weight index, tokenizer) identifies the snapshot even
where no VCS commit is available. Sharded weight files are hashed individually: the weight
index maps parameters to shards but carries no content hashes, so without shard digests a
partial, corrupted, or substituted download could share a manifest with the genuine
snapshot."""
from __future__ import annotations
import argparse, datetime, glob, hashlib, json, os, pathlib

FILES = ("config.json", "generation_config.json", "model.safetensors.index.json",
         "model.safetensors", "tokenizer.json", "tokenizer_config.json", "vocab.json")


def _sha(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def resolve(model: str, cache_dir: str):
    if os.path.isdir(model):
        return pathlib.Path(model), None  # local dir (e.g. ModelScope download): no VCS rev
    snaps = glob.glob(os.path.join(cache_dir, f"models--{model.replace('/', '--')}",
                                   "snapshots", "*"))
    snaps = [s for s in snaps if os.path.isdir(s)]
    if len(snaps) != 1:
        raise SystemExit(f"expected exactly one cached snapshot for {model}, found {snaps}")
    return pathlib.Path(snaps[0]), os.path.basename(snaps[0])  # dirname IS the hub commit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model id (HF cache) or local directory")
    ap.add_argument("--cache-dir", default="models/huggingface")
    ap.add_argument("--out", default="results/model_snapshots.jsonl")
    args = ap.parse_args()
    d, rev = resolve(args.model, args.cache_dir)
    manifest = {os.path.relpath(str(p), str(d)): _sha(p)
                for name in FILES for p in [d / name] if p.exists()}
    for p in sorted(d.glob("model-*-of-*.safetensors")):
        manifest[p.name] = _sha(p)
    if not manifest:
        raise SystemExit(f"no identifying files found under {d}")
    rec = {"model": args.model, "dir": str(d), "revision": rev,
           "recorded": datetime.datetime.now().isoformat(timespec="seconds"),
           "sha256": manifest}
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"{args.model}: revision={rev or 'n/a (local dir; identified by sha256 manifest)'}, "
          f"{len(manifest)} files hashed -> {args.out}")


if __name__ == "__main__":
    main()
