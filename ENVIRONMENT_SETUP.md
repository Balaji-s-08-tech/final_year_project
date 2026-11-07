# Bunny Phi-2 environment setup

The existing local environment is `.venv-vlm-awq` (Python 3.11.9). It already contains PyTorch 2.2.2+cu121, torchvision 0.17.2+cu121, Transformers 4.37.2, Tokenizers 0.15.1, Accelerate 0.28.0, PEFT 0.10.0, SentencePiece 0.1.99, NumPy 1.26.4, FastAPI 0.115.12, Uvicorn 0.34.2, and psutil 7.2.2. CUDA is available on the NVIDIA RTX 3050 Laptop GPU (6 GB dedicated VRAM). This run did not create a second environment or reinstall the ML stack.

The Bunny model path is Phi-2. Keep Transformers pinned to 4.37.2 for compatibility with the local modeling code. The small dashboard dependencies are listed separately in `requirements-dashboard.txt`; the ML environment pins are in `requirements-bunny-awq.txt`. Install the CUDA PyTorch wheel separately, using the compatible CUDA wheel index and version for your driver rather than allowing the broad `torch>=2.0` requirement to select an arbitrary build.

## Verify the existing environment

```powershell
.\.venv-vlm-awq\Scripts\python.exe -c "import torch, transformers, tokenizers, accelerate, peft, sentencepiece, numpy; print(torch.__version__, torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'); print(transformers.__version__, tokenizers.__version__, accelerate.__version__, peft.__version__, sentencepiece.__version__, numpy.__version__)"
```

Import checks from the repository root:

```powershell
.\.venv-vlm-awq\Scripts\python.exe -c "import sys; sys.path.insert(0, 'LLM-Pruner'); import LLMPruner; print('LLMPruner import: OK')"
.\.venv-vlm-awq\Scripts\python.exe -c "import sys; sys.path.insert(0, 'VLM'); from bunny.model.builder import load_pretrained_model; print('Bunny builder import: OK')"
```

No full Bunny checkpoint download was performed. The local Hugging Face cache currently has the Bunny config and tokenizer assets but not the model weight files. A 3B model load and recovery are therefore blocked until the weights and project recovery data are available. At the time of audit, system RAM availability was about 4 GiB, which is insufficient headroom for safely expanding the existing portable full-model smoke artifact.

## Run focused tests

```powershell
.\.venv-vlm-awq\Scripts\python.exe -m pytest backend/tests VLM/quantization/tests -q
.\.venv-vlm-awq\Scripts\python.exe -m compileall -q VLM/quantization backend
.\.venv-vlm-awq\Scripts\python.exe -m VLM.quantization.benchmark --help
```

## Local dashboard (optional)

Install `requirements-dashboard.txt`, start `python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000`, then run `npm install` and `npm run dev` in `D:\final year project\Interface`. Vite defaults to port 3000, matching the backend CORS default. The dashboard worker runs one job at a time and defaults to CPU for AWQ.
