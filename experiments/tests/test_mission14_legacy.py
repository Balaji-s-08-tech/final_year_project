import ast
import json
from pathlib import Path
import pytest
from experiments.store import read_json
ROOT=Path(__file__).resolve().parents[2]
LEGACY=('ShortGPT/short_gpt/prune.py', 'ShortGPT/short_gpt/short_hf.py', 'ShortGPT/short_gpt/short_vlm.py', 'ShortGPT/short_gpt/short_llama.py', 'LLM-Pruner/examples/InternVL.py', 'LLM-Pruner/examples/Bunny.py', 'LLM-Pruner/examples/llava-vicuna_prune.py', 'LLM-Pruner/post_training.py', 'VLM/bunny/train/train.py', 'VLM/bunny/train/train_pruned.py')
@pytest.mark.parametrize('relative',LEGACY)
def test_historical_routes_quarantined_before_imports(relative):
    tree=ast.parse((ROOT/relative).read_text())
    executable=[n for n in tree.body if not (isinstance(n,ast.Expr) and isinstance(n.value,ast.Constant) and isinstance(n.value.value,str)) and not (isinstance(n,ast.ImportFrom) and n.module=='__future__')]
    assert isinstance(executable[0],ast.Raise),'Unreviewed historical route lacks fail-closed module admission'
