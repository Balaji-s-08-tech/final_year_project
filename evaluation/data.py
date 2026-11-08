from experiments.jsonio import loads as strict_json_loads
"""Versioned exact-match JSONL protocol. No implicit data downloads."""
import hashlib
import json
import random
import unicodedata
from pathlib import Path
from experiments.provenance import canonical_hash, file_hash


def normalize(answer: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", answer).casefold().split())


def load_examples(config, *, max_hash_bytes=None):
    path = Path(config.dataset_path).resolve()
    if max_hash_bytes is not None and file_hash(path,max_hash_bytes).get('status')!='HASHED':
        raise ValueError('Dataset exceeds explicit verification budget or is unavailable')
    digest = hashlib.sha256()
    records, identifiers = [], set()
    from experiments.jsonio import lines
    for line in lines(path):
        digest.update(line)
        if not line.strip():
            continue
        record = strict_json_loads(line)
        if not isinstance(record, dict) or set(record) - {"id", "question", "answers", "image"}:
            raise ValueError("Evaluation records must contain only id/question/answers/image")
        if not isinstance(record.get("id"), str) or not record["id"] or record["id"] in identifiers:
            raise ValueError("Example IDs must be nonempty and unique")
        identifiers.add(record["id"])
        if not isinstance(record.get("question"), str) or not record["question"].strip() or "<image>" in record["question"]:
            raise ValueError("Question must be nonempty text without image sentinels")
        if not isinstance(record.get("answers"), list) or not record["answers"] or not all(isinstance(a, str) and normalize(a) for a in record["answers"]):
            raise ValueError("Every example requires nonempty reference answers")
        if config.adapter == "bunny_phi":
            if not isinstance(record.get("image"), str):
                raise ValueError("Multimodal examples require local images")
            image = (path.parent / record["image"]).resolve()
            if not image.is_relative_to(path.parent) or not image.is_file():
                raise ValueError("Image must exist within the dataset directory")
            record["image"] = str(image)
        elif record.get("image") is not None:
            raise ValueError("Text-only adapter cannot silently discard images")
        records.append(record)
    if len(records) < config.max_samples:
        raise ValueError("Fewer examples than max_samples; refusing a silent sample-count change")
    chosen = random.Random(config.seed).sample(range(len(records)), config.max_samples)
    selected = [records[i] for i in chosen]
    images={}
    for record in selected:
        if record.get('image'):
            evidence=file_hash(record['image'],max_hash_bytes)
            if evidence.get('status')!='HASHED':raise ValueError('Selected image unavailable or exceeds verification budget')
            images[record['id']]=evidence['sha256']
    return selected, canonical_hash({"jsonl_sha256": digest.hexdigest(), "selected_image_sha256": images})
