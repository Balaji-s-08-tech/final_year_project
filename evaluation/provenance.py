"""Bound small assets and record effective settings; large weights are explicitly skipped."""
from pathlib import Path
from experiments.provenance import file_hash, canonical_hash


def asset_manifest(config):
    manifest = {}
    for region, source in (("model", config.model_path), ("encoder", config.vision_path)):
        if not source:
            manifest[region] = {"status": "UNAVAILABLE"}
            continue
        root = Path(source)
        paths = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
        files = {p.name if root.is_file() else str(p.relative_to(root)): file_hash(p) for p in paths}
        manifest[region] = {"path": str(root), "files": files, "status": "CONTENT_HASHED" if files and all(v["status"] == "HASHED" for v in files.values()) else "INCOMPLETE"}
    manifest["sha256"] = canonical_hash(manifest)
    return manifest
