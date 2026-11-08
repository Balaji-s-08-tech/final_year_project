"""Bound small assets and record effective settings; large weights are explicitly skipped."""
from pathlib import Path
from experiments.provenance import canonical_hash
from experiments.assets import inventories, DEFAULT_ASSET_LIMITS


def asset_manifest(config, *, limits=DEFAULT_ASSET_LIMITS):
    manifest = {}
    sources = [(region, source) for region, source in (("model", config.model_path), ("encoder", config.vision_path))]
    discovered = iter(inventories([source for _, source in sources if source], hash_large=getattr(config, "hash_large_assets", False), limits=limits))
    for region, source in sources:
        if not source:
            manifest[region] = {"status": "UNAVAILABLE"}
            continue
        root = Path(source)
        files = next(discovered)
        manifest[region] = {"path": str(root), "files": files, "status": "CONTENT_HASHED" if files and all(v["status"] == "HASHED" for v in files.values()) else "INCOMPLETE"}
    manifest["sha256"] = canonical_hash(manifest)
    return manifest
