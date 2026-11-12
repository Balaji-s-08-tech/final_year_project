from dataclasses import replace
import pytest
import torch

from evaluation.tests.test_baseline import tiny_bunny
from evaluation.adapters import LocalAdapter
from evaluation.data import load_examples
from compression.pruning import prune, PruningConfig
from compression.multimodal import base_identity, protected_identity, export_treatment


@pytest.mark.parametrize('kind', ['dense', 'mask', 'pruned', 'recovered', 'int4', 'combined'])
def test_complete_bunny_treatment_reload_and_visual_generation(tiny_bunny, tmp_path, kind):
    adapter = LocalAdapter()
    adapter.load(tiny_bunny)
    source = base_identity(tiny_bunny)
    protected = protected_identity(adapter.model)
    widths = [b.mlp.fc1.out_features for b in adapter.model.model.layers]
    recovery = None
    if kind in ('pruned', 'combined', 'recovered'):
        prune(adapter.model, PruningConfig(method='mlp_neurons', sparsity=.25))
    if kind == 'mask':
        prune(adapter.model, PruningConfig(sparsity=.25))
    if kind == 'recovered':
        from compression.recovery import recover_lora
        ids = torch.tensor([[4, 5, 6]])
        adapted, recovery = recover_lora(adapter.model, [{'input_ids': ids, 'attention_mask': torch.ones_like(ids), 'labels': ids}])
        adapter.model = adapted.merge_and_unload(safe_merge=True).eval()
    if kind in ('int4', 'combined'):
        from VLM.quantization.awq_quantize import quantize_bunny_awq
        from VLM.quantization.awq_config import AWQConfig
        quantize_bunny_awq(adapter.model, [torch.tensor([[4, 5, 6]])], AWQConfig(
            calibration_samples=1, scale_grid=2, clip_grid=2, group_size=8))
    folder = tmp_path / kind
    document = export_treatment(adapter, folder, source_identity=source, original_widths=widths,
                                protected_sha256=protected, recovery=recovery)
    config = replace(tiny_bunny, treatment_artifact_path=str(folder), treatment=document['treatment'])
    restored = LocalAdapter()
    restored.load(config)
    assert protected_identity(restored.model) == protected
    record = load_examples(config)[0][0]
    with torch.inference_mode():
        original = adapter.generate(adapter.prepare(record))
        result = restored.generate(restored.prepare(record))
    assert torch.equal(original, result)
    assert restored.decode(restored.prepare(record), result).token_ids
    assert restored.effective()['asset_verification']['treatment']['sha256'] == document['sha256']


def test_dense_cannot_be_labeled_compressed(tiny_bunny, tmp_path):
    from experiments.store import read_json
    import json
    adapter = LocalAdapter(); adapter.load(tiny_bunny)
    folder = tmp_path / 'dense'
    export_treatment(adapter, folder, source_identity=base_identity(tiny_bunny), original_widths=[32],
                     protected_sha256=protected_identity(adapter.model))
    path = folder / 'treatment.json'
    document = read_json(path); document['treatment'] = 'structured_pruning'
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='label contradicts'):
        LocalAdapter().load(replace(tiny_bunny, treatment_artifact_path=str(folder), treatment='structured_pruning'))


@pytest.mark.parametrize('damage', ['base','weights','coverage','protected','declaration'])
def test_treatment_tampering_rejected(tiny_bunny,tmp_path,damage):
    import json
    from experiments.store import read_json
    adapter=LocalAdapter();adapter.load(tiny_bunny)
    folder=tmp_path/'artifact'
    export_treatment(adapter,folder,source_identity=base_identity(tiny_bunny),original_widths=[32],protected_sha256=protected_identity(adapter.model))
    document=read_json(folder/'treatment.json')
    if damage=='base':document['base_sha256']='0'*64
    elif damage=='protected':document['protected_sha256']='0'*64
    elif damage=='coverage':document['linears'].pop(next(iter(document['linears'])))
    elif damage=='declaration':document['linears'][next(iter(document['linears']))]=[]
    else:
        with (folder/'blocks.safetensors').open('ab') as stream:stream.write(b'bad')
    (folder/'treatment.json').write_text(json.dumps(document))
    with pytest.raises(ValueError):LocalAdapter().load(replace(tiny_bunny,treatment_artifact_path=str(folder)))
