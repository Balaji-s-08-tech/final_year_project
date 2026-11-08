"""Local, explicit content manifests. No asset acquisition or remote clients."""
from pathlib import Path
from experiments.provenance import file_hash, canonical_hash


def model_manifest(root, identifier, revision, *, hash_large=False):
    root = Path(root).resolve()
    if not root.is_dir() or not isinstance(identifier, str) or not identifier.strip() or not isinstance(revision, str) or not revision.strip() or revision.startswith("REPLACE_"):
        raise ValueError("Pinned local model identity/revision and directory are required")
    files = {}
    for file in sorted(root.rglob('*')):
        if file.is_file():
            if not file.resolve().is_relative_to(root):
                raise ValueError("Asset symlink escapes its root")
            files[str(file.relative_to(root))] = file_hash(file, None if hash_large else 8*1024*1024)
    if not files:
        raise ValueError("Empty asset directory")
    return {"schema_version": 1, "kind": "model", "identifier": identifier, "revision": revision,
            "files": files, "status": "VERIFIED_LOCAL_CONTENT" if all(v['status']=='HASHED' for v in files.values()) else "INCOMPLETE"}


def verify_model_manifest(root, document, *, hash_large=False):
    if document.get('schema_version') != 1 or document.get('kind') != 'model':
        raise ValueError("Unsupported manifest")
    actual = model_manifest(root, document['identifier'], document['revision'], hash_large=hash_large)
    if actual['files'] != document['files'] or actual['status'] != 'VERIFIED_LOCAL_CONTENT':
        raise ValueError("Asset content differs or remains unverified")
    return actual


def dataset_manifest(config, *, split, purpose):
    from evaluation.data import load_examples
    if not split or purpose not in {'evaluation', 'calibration', 'recovery'}:
        raise ValueError("Explicit split and data purpose required")
    records, fingerprint = load_examples(config)
    return {"schema_version": 1, "kind": "dataset", "identifier": config.dataset_identifier,
            "revision": config.dataset_revision, "split": split, "purpose": purpose,
            "fingerprint": fingerprint, "sample_ids": [r['id'] for r in records],
            "image_sha256": [file_hash(r['image'], None)['sha256'] for r in records if r.get('image')]}


def require_disjoint(evaluation, calibration):
    if evaluation.get('purpose') != 'evaluation' or calibration.get('purpose') not in {'calibration', 'recovery'}:
        raise ValueError("Data purposes must be explicit")
    if set(evaluation['sample_ids']) & set(calibration['sample_ids']) or set(evaluation.get('image_sha256', [])) & set(calibration.get('image_sha256', [])):
        raise ValueError("Evaluation/calibration overlap detected")
    if evaluation['fingerprint'] == calibration['fingerprint']:
        raise ValueError("Evaluation/calibration fingerprints are identical")
    return {"status": "CHECKED_SELECTED_IDS_AND_IMAGES", "warning": "This does not prove absence of semantic duplicates or training contamination"}
