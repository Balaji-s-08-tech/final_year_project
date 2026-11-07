# Local research dashboard

The React dashboard in the sibling `Interface` directory reads the local FastAPI service in `backend/`. It shows registered artifacts, calibration text files, system resources, job state and benchmark manifests. It has no synthetic fallback data. If the service is unreachable, it shows a disconnected state.

## Research pipeline and current boundary

The underlying research pipeline is structured pruning, recovery training, then post-training quantization. AWQ calibration and scales must be computed from the exact checkpoint supplied to the quantizer, after pruning and recovery. This dashboard does not run pruning or recovery: the pruning scripts remain separate, and no recovery worker is configured. Therefore, users must supply the intended pruned and recovered checkpoint explicitly; a filename alone is not proof of recovery lineage.

The API registers files found under `smoke_test_output/`, `test_awq_output/`, `checkpoints/`, and optional project-local directories listed in `VLM_CHECKPOINT_DIRS`. Checkpoint size is measured from the file. Parameter count is not inferred. The simple `AWQ output` lineage label is filename-based and does not prove run provenance. The current registered AWQ format supports text generation through its portable PyTorch layer implementation. Image inference is not supported by that language-only artifact.

AWQ processing reuses the project quantization integration and core: calibration samples are selected reproducibly, activations are collected on the selected checkpoint, scale search and clipping run before weight packing, and the result is saved with configuration and metadata. The dashboard submits four samples with sequence length 32 to keep the default operation bounded. Its default execution device is CPU; CUDA submission is rejected unless the runtime reports at least 5 GiB free VRAM. The portable linear layer dequantizes during PyTorch forward and is not an optimized CUDA kernel.

## Start the backend

From the repository root, in PowerShell:

```powershell
.\.venv-vlm-awq\Scripts\Activate.ps1
python -m pip install -r requirements-dashboard.txt
$env:VLM_FRONTEND_ORIGIN = 'http://localhost:5173'
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

The CORS origin defaults to `http://localhost:3000`; set it as above for Vite's default port. If Vite selects another port, set `VLM_FRONTEND_ORIGIN` to that origin before starting the backend.

## Start the frontend

In a second PowerShell window:

```powershell
Set-Location 'D:\final year project\Interface'
npm install
npm run dev
```

Set `VITE_API_BASE_URL` in the frontend environment to the backend URL if it is not `http://localhost:8000`.

## Available API operations

- `GET /api/health`, `/api/system/info`, `/api/models`, `/api/checkpoints`, `/api/calibration-data`
- `GET /api/jobs`, `/api/jobs/{id}`, `/api/jobs/{id}/logs`; `POST /api/jobs`; `POST /api/jobs/{id}/cancel`
- `POST /api/inference` for text-only inference on a registered portable AWQ checkpoint
- `GET /api/benchmarks`, `POST /api/results/import`, and `/api/pipeline/capabilities`

Jobs run one at a time in a local worker. The API accepts quantize, artifact-size benchmark, and text-inference jobs. It does not accept arbitrary shell commands, model-weight uploads, pruning, recovery, image inference, or cloud jobs. The Benchmarks page imports a JSON manifest from the user's local computer; result values and checkpoint metadata are stored with an import timestamp/source label and displayed as supplied. This does not independently verify a remote run. Calibration and checkpoint paths must resolve inside approved project directories. Use `VLM_CHECKPOINT_DIRS` to add project-local directories.

## Validation and research metrics

Focused tests:

```powershell
.\.venv-vlm-awq\Scripts\python.exe -m pytest backend/tests VLM/quantization/tests -q
```

The benchmark harness measures artifact size only. Peak GPU memory, generation latency, tokens per second, and task accuracy remain `NOT MEASURED` until connected to a controlled evaluation protocol. INT4 and INT8 results must be compared with precision reported explicitly; the dashboard makes no claim that AWQ INT4 is better than INT8.

Current project smoke artifacts include a pruning output and a portable AWQ output. The existing smoke run skipped recovery due to available hardware limits. Their presence does not establish a completed pruning → recovery → AWQ experimental result. Full model quality evaluation and a recovery checkpoint validated for lineage remain pending.
