"""Optional local image-question API. No model load until an explicit request."""
import os
import re
from pathlib import Path
import threading
import time
from dataclasses import replace

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from experiments.store import read_json
from evaluation.config import EvaluationConfig

router = APIRouter(prefix='/api/demo')


class ImageQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid', strict=True)
    model_id: str = Field(min_length=1, max_length=80, pattern=r'^[A-Za-z0-9_-]+$')
    image_id: str = Field(min_length=1, max_length=200)
    question: str = Field(min_length=1, max_length=1000)
    max_new_tokens: int = Field(default=32, ge=1, le=64)


class DemoService:
    def __init__(self):
        self.lock = threading.Lock()

    def registry(self):
        filename = os.environ.get('VLM_DEMO_REGISTRY')
        if not filename:
            return {}
        document = read_json(filename)
        if not isinstance(document, dict) or len(document) > 8:
            raise ValueError('Registry must contain at most eight model configurations')
        if any(not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,80}', key) or not isinstance(value, str) or not value.strip()
               for key, value in document.items()):
            raise ValueError('Registry requires safe model IDs and local config filenames')
        configs = {key: EvaluationConfig.load(Path(filename).parent / value) for key, value in document.items()}
        for config in configs.values():
            if config.adapter != 'bunny_phi' or config.max_samples > 128 or config.max_input_tokens > 2048 or config.max_new_tokens > 64:
                raise ValueError('Live image demo requires bounded Bunny configurations')
        return configs

    def catalog(self):
        from evaluation.data import load_examples
        return [{'id': key, 'treatment': cfg.treatment, 'identity_status': 'CONFIGURED_NOT_LOADED',
                 'model_identifier': cfg.model_identifier, 'model_revision': cfg.model_revision,
                 'images': [{'id': row['id'], 'question': row['question']} for row in load_examples(cfg)[0]]}
                for key, cfg in self.registry().items()]

    def infer(self, request):
        if not request.question.strip() or '<image>' in request.question or '\x00' in request.question:
            raise ValueError('Provide a text question without image markers or NUL')
        configs = self.registry()
        if request.model_id not in configs:
            raise KeyError('Unknown registered model')
        config = configs[request.model_id]
        from evaluation.data import load_examples
        records = load_examples(config)[0]
        record = next((r for r in records if r['id'] == request.image_id), None)
        if record is None:
            raise KeyError('Unknown registered image')
        if not self.lock.acquire(blocking=False):
            raise RuntimeError('Another inference is running')
        try:
            import torch
            from evaluation.adapters import LocalAdapter
            from runtime_policy import admit_device
            authorized = os.environ.get('VLM_DEMO_ALLOW_GPU') == '1'
            admit_device(config.device, authorized)
            if config.device.startswith('cuda'):
                from evaluation.preflight import preflight
                report = preflight(config)
                if report['errors']:
                    raise ValueError('GPU preflight blocked: ' + '; '.join(report['errors']))
            config = replace(config, max_new_tokens=min(config.max_new_tokens, request.max_new_tokens))
            adapter = LocalAdapter(); adapter.authorize_gpu = authorized
            if config.device.startswith('cuda'):
                torch.cuda.reset_peak_memory_stats(config.device)
            adapter.load(config)
            with torch.inference_mode():
                prepared = adapter.prepare({**record, 'question': request.question})
                if config.device.startswith('cuda'): torch.cuda.synchronize(config.device)
                start = time.perf_counter()
                result = adapter.decode(prepared, adapter.generate(prepared))
                if config.device.startswith('cuda'): torch.cuda.synchronize(config.device)
                elapsed = time.perf_counter() - start
            if not result.token_ids:
                raise ValueError('Model generated no continuation')
            return {'answer': result.text, 'generated_tokens': len(result.token_ids), 'latency_seconds': elapsed,
                'peak_vram_bytes': torch.cuda.max_memory_allocated(config.device) if config.device.startswith('cuda') else None,
                'identity': {'id': request.model_id, 'identifier': config.model_identifier, 'revision': config.model_revision,
                    'treatment': config.treatment, 'artifact_sha256': (getattr(adapter, 'treatment_identity', None) or {}).get('sha256'),
                    'weight_origin': adapter.weight_origin, 'verification': 'LOADED_TENSORS_VALIDATED'},
                'compression': {k: v for k, v in (getattr(adapter, 'treatment_identity', None) or {}).items()
                                if k in ('physical_pruning','masked','packed_bits','recovery')},
                'architecture': adapter.inspect()}
        finally:
            self.lock.release()


service = DemoService()


@router.get('/models')
def catalog():
    try:
        return {'models': service.catalog(), 'status': 'optional_local_backend'}
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise HTTPException(503, 'Local model registry is unavailable or invalid') from exc


@router.post('/inference')
def inference(request: ImageQuestion):
    try:
        return service.infer(request)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except (ValueError, OSError) as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
