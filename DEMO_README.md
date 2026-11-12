# Offline final-year demonstration

Build once with Node/npm and then run entirely offline, without Camber, model
weights, a backend or CUDA:

```sh
cd Interface
npm ci
npm run lint
npm run build
cd ..
python scripts/launch_demo.py
```

Open `http://localhost:3000`. The default presentation includes architecture,
interactive paired-neuron pruning and INT4 explanations, LoRA, research treatment
comparison, a stored synthetic image/question and provenance. The engineering
console remains accessible. The optional backend is contacted only when the user
connects local models or opens that console.

On Windows PowerShell the same commands work with installed Python and Node.
No Bash activation is required. Use `py scripts/launch_demo.py` if Python is
registered through the Windows launcher. Stop with Ctrl+C. A prebuilt `dist`
directory can be copied to an offline machine; generated builds are not in Git.
Native Windows execution has not been performed in this Linux environment.

The main research table intentionally shows unavailable full-model metrics. A
separate table contains actual measurements from nine tiny random **text-only** CPU
fixtures, labeled as diagnostic teaching evidence. Duration includes preparation
and evaluation; it is not inference latency. The snapshot retains source commit
and dirty-source disclosure. No synthetic score represents Bunny-3B quality.

## Optional real local inference

Create a private local `registry.json` mapping stable IDs to evaluation config files:

```json
{"dense":"dense.json","compressed":"compressed.json"}
```

These configs use staged Bunny/SigLIP/tokenizer assets and a local image dataset.
The compressed config additionally selects a verified treatment artifact as
described in MODEL_ARTIFACT_MANIFEST.md. No paths come from HTTP requests.

Linux shell:

```sh
export VLM_DEMO_REGISTRY=/absolute/path/registry.json
python scripts/launch_demo.py --backend
```

PowerShell: `$env:VLM_DEMO_REGISTRY='C:\assets\registry.json'`, then run the launcher
using the ML environment interpreter (for example `.venv-cpu\Scripts\python.exe`).
For separately authorized CUDA inference also set `VLM_DEMO_ALLOW_GPU=1`; the
config must declare device/dtype, held-out split and RAM/VRAM budgets. The API is
loopback-only and single-inference-at-a-time. It loads the selected model per
request; reported latency covers generation, not loading. Registry entries are
declared identities until tensor loading validates them. Returned identities include
the actual treatment hash and compression layout; a dense artifact relabeled as
compressed is rejected.

## Browser regression

With the built demo serving on port 3000, run `npm run test:demo` from Interface.
Playwright is pinned as a development dependency. Set `CHROMIUM_PATH` to
an installed Chromium binary when needed. The test blocks all API calls and checks
navigation, interaction, metrics, image loading and disconnected inference, then
uses an explicitly mocked response to test the live UI contract. The actual API
is independently tested using complete CPU Bunny fixtures. If no browser exists,
install the matching browser with `npx playwright install chromium` on the test
machine while connected; serving the offline demonstration does not need Playwright.
