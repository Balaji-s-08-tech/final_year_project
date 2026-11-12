"""Stage pinned public assets explicitly, prepare a smoke config, or run it locally.

No downloads or GPU work happen on import, --help, or prepare.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

BUNNY = ('BAAI/Bunny-v1_0-3B', '0958a104bd39a448fc310b3d1b1fc3d398dc3fcb')
SIGLIP = ('google/siglip-so400m-patch14-384', '9fdffc58afc957d1a03a25b10dba0329ab15c2a3')
BUNNY_FILES = ['config.json', 'generation_config.json', 'model.safetensors.index.json',
    'model-00001-of-00002.safetensors', 'model-00002-of-00002.safetensors',
    'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
    'added_tokens.json', 'vocab.json', 'merges.txt']
SIGLIP_FILES = ['config.json', 'model.safetensors', 'preprocessor_config.json']


def download(destination, authorized=False):
    if authorized is not True:
        raise PermissionError('Asset download requires --authorize-download and reviewed storage/network access')
    root = Path(destination).resolve()
    # Reviewed source files total approximately 9.9 GB; leave room for staging.
    parent = root.parent
    if not parent.is_dir() or shutil.disk_usage(parent).free < 16 * 1024**3:
        raise ValueError('Create the destination parent and provide at least 16 GiB free staging space')
    root.mkdir(exist_ok=False)
    from huggingface_hub import snapshot_download
    from experiments.store import write_json
    for name, identity, files in [('bunny', BUNNY, BUNNY_FILES), ('siglip', SIGLIP, SIGLIP_FILES)]:
        snapshot_download(repo_id=identity[0], revision=identity[1],
                          local_dir=str(root / name), allow_patterns=files)
    write_json(root/'acquisition.json', {'bunny': BUNNY, 'siglip': SIGLIP,
        'status': 'PINNED_HUB_DOWNLOAD_COMPLETED', 'authenticity': 'DEPENDENT_ON_HUB_SOURCE_AND_LOCAL_STORAGE'})


def resource_review(config):
    from evaluation.inspection import local_checkpoint
    import psutil
    model, vision = local_checkpoint(config.model_path), local_checkpoint(config.vision_path)
    sizes = [model['artifact_bytes'], vision['artifact_bytes']]
    total = sum(sizes) if all(type(size) is int for size in sizes) else None
    ram = psutil.virtual_memory().available
    return {'model_weight_file_bytes': sizes[0], 'vision_weight_file_bytes': sizes[1],
        'staged_weight_file_bytes': total, 'available_ram_bytes': ram,
        'available_disk_bytes': shutil.disk_usage(Path(config.model_path)).free,
        'existing_cpu_preparation_ram_gate_bytes': None if total is None else total * 8,
        'passes_existing_cpu_preparation_ram_gate': total is not None and ram >= total * 8,
        'compressed_gpu_fit': 'NOT_PROVEN',
        'warning': 'Baseline smoke memory does not establish pruning, LoRA, AWQ, export-copy or checkpoint-space feasibility'}


def prepare(model_dir, vision_dir, output, acknowledged=False):
    if acknowledged is not True:
        raise PermissionError('Confirm that local assets came from the pinned revisions with --acknowledge-pinned-assets')
    from evaluation.config import EvaluationConfig
    from evaluation.preflight import preflight
    from experiments.manifests import model_manifest
    from experiments.store import write_json
    from PIL import Image, ImageDraw
    model_dir, vision_dir, root = (Path(p).resolve() for p in (model_dir, vision_dir, output))
    if any(root.is_relative_to(source) for source in (model_dir, vision_dir)):
        raise ValueError('Preparation output must be outside both asset directories')
    if root.exists():
        raise FileExistsError('Preparation output already exists; choose a new directory')
    # Hash existing local assets, never relabel a previous experiment or overwrite output.
    manifests = [model_manifest(source, *identity, hash_large=True)
                 for source, identity in ((model_dir, BUNNY), (vision_dir, SIGLIP))]
    root.mkdir(parents=True, exist_ok=False)
    for filename, document in zip(('bunny_manifest.json', 'siglip_manifest.json'), manifests):
        write_json(root/filename, document)
    image = Image.new('RGB', (384, 384), 'white')
    ImageDraw.Draw(image).rectangle((110, 110, 274, 274), fill='red')
    image.save(root/'red_square.png')
    sample = {'id': 'smoke-red-square-001', 'question': 'What color is the square?',
              'answers': ['red'], 'image': 'red_square.png'}
    # Exclusive publication; the generated sample is teaching data, not a benchmark.
    from experiments.store import publish
    publish(root/'samples.jsonl', lambda stream: stream.write((json.dumps(sample)+'\n').encode()))
    config = EvaluationConfig(str(model_dir), *BUNNY, str(root/'samples.jsonl'),
        'local-synthetic-smoke-test', 'generated-red-square-v1',
        'USER: <image>\n{question}\nASSISTANT:', vision_path=str(vision_dir),
        model_manifest_path=str(root/'bunny_manifest.json'), vision_manifest_path=str(root/'siglip_manifest.json'),
        device='cuda:0', dtype='float16', max_samples=1, max_input_tokens=128,
        max_new_tokens=16, warmup_iterations=0, dataset_split='synthetic-smoke-only',
        required_free_ram_bytes=20*1024**3, required_free_vram_bytes=10*1024**3,
        hash_large_assets=True, evaluation_purpose='smoke_test')
    write_json(root/'smoke_config.json', config.to_dict())
    write_json(root/'source_assertion.json', {'bunny': BUNNY, 'siglip': SIGLIP,
        'acquisition_provenance': 'OPERATOR_ASSERTED_PINNED_LOCAL_ASSETS',
        'note': 'Content hashes prove local consistency, not remote source authenticity'})
    write_json(root/'resource_review.json', resource_review(config))
    write_json(root/'preflight.json', preflight(config))
    return root/'smoke_config.json'


def run(config_path, output, authorized=False):
    # This assignment precedes the first import that could import torch.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    from evaluation.config import EvaluationConfig
    from evaluation.runner import evaluate
    from experiments.store import read_json
    config = EvaluationConfig.load(config_path)
    if (config.model_identifier, config.model_revision) != BUNNY or config.evaluation_purpose != 'smoke_test' or config.treatment != 'baseline' or config.treatment_artifact_path:
        raise ValueError('Reproduction requires the pinned dense Bunny smoke configuration')
    vision = read_json(config.vision_manifest_path)
    if (vision.get('identifier'), vision.get('revision')) != SIGLIP:
        raise ValueError('Reproduction requires the pinned SigLIP manifest')
    return evaluate(config, output, authorize_gpu=authorized).path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    acquire = commands.add_parser('download')
    acquire.add_argument('--destination', required=True)
    acquire.add_argument('--authorize-download', action='store_true')
    stage = commands.add_parser('prepare')
    stage.add_argument('--model-dir', required=True)
    stage.add_argument('--vision-dir', required=True)
    stage.add_argument('--output', required=True)
    stage.add_argument('--acknowledge-pinned-assets', action='store_true')
    execute = commands.add_parser('run')
    execute.add_argument('--config', required=True)
    execute.add_argument('--output', required=True)
    execute.add_argument('--authorize-gpu-execution', action='store_true')
    args = parser.parse_args()
    if args.command == 'download':
        download(args.destination, args.authorize_download)
    elif args.command == 'prepare':
        print(prepare(args.model_dir, args.vision_dir, args.output, args.acknowledge_pinned_assets))
    else:
        print(run(args.config, args.output, args.authorize_gpu_execution))


if __name__ == '__main__': main()
