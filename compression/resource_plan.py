"""Read-only tensor accounting, not an execution admission override or fit proof."""
import argparse
import json
import math


def estimate(hidden, intermediate, layers, parameters, *, sparsity=.25, group_size=128, rank=2):
    for name, value in [('hidden',hidden),('intermediate',intermediate),('layers',layers),
                        ('parameters',parameters),('group_size',group_size),('rank',rank)]:
        if type(value) is not int or not 1 <= value <= 2**63-1:
            raise ValueError(f'Positive integer {name} required')
    if type(sparsity) not in (int,float) or not math.isfinite(sparsity) or not 0 <= sparsity < 1:
        raise ValueError('Finite sparsity in [0,1) required')
    kept = intermediate - int(intermediate*sparsity)
    if rank > min(hidden, kept):
        raise ValueError('LoRA rank exceeds supported target dimensions')
    dense_weights = layers*(4*hidden**2 + 2*hidden*intermediate)
    # Phi: four attention biases, fc1/fc2 biases, input LayerNorm weight/bias.
    dense_other = layers*(7*hidden+intermediate)
    if parameters < dense_weights+dense_other:
        raise ValueError('Parameter count smaller than declared Phi blocks')
    removed = layers*(intermediate-kept)*(2*hidden+1)
    pruned_parameters = parameters-removed
    pruned_weights = layers*(4*hidden**2+2*hidden*kept)
    pruned_other = layers*(7*hidden+kept)
    groups = layers*(4*hidden*((hidden+group_size-1)//group_size) + kept*((hidden+group_size-1)//group_size)
                     + hidden*((kept+group_size-1)//group_size))
    packed = layers*(4*hidden*((hidden+1)//2)+kept*((hidden+1)//2)+hidden*((kept+1)//2))
    adapters = layers*rank*2*(hidden+kept)
    source_pair = 4*(2*hidden*intermediate+intermediate+hidden)
    target_pair = 4*(2*hidden*kept+kept+hidden)
    return {'status':'ANALYTICAL_TENSOR_ACCOUNTING_NOT_MEASURED_PEAK',
        'kept_neurons_per_layer':kept,
        'dense_fp32_model_bytes':4*parameters,
        'dense_block_fp32_payload_bytes':4*(dense_weights+dense_other),
        'pruned_fp32_model_bytes':4*pruned_parameters,
        'pruned_block_fp32_payload_bytes':4*(pruned_weights+pruned_other),
        'original_mlp_map_retention_bytes':4*layers*(2*hidden*intermediate+intermediate+hidden),
        'physical_score_temporary_bytes':16*hidden*intermediate,
        'single_pruned_mlp_bytes':target_pair,
        'lora_fp32_source_plus_copy_bytes':8*pruned_parameters,
        'lora_adapter_elements':adapters,
        'lora_adapter_grad_adam_bytes':16*adapters,
        'awq_pruned_block_int4_payload_bytes':packed+5*groups+4*pruned_other,
        'awq_pruned_fp32_protected_model_bytes':4*(pruned_parameters-pruned_weights)+packed+5*groups,
        'awq_64_token_feature_bytes':64*(5*hidden+kept)*4,
        'one_mlp_pilot':{'source_fp32_bytes':source_pair,'pruned_fp32_bytes':target_pair,
                         'tensor_peak_planning_bytes':source_pair+max(16*hidden*intermediate,3*target_pair)},
        'excludes':['nonparameter buffers','Python/runtime','allocator fragmentation','mapped/page-cached source files',
                    'LoRA training activations','unmeasured serializer buffers','CUDA context/workspaces'],
        'warning':'Do not use these tensor counts as measured RSS/VRAM or to bypass existing execution gates'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hidden',type=int,required=True)
    parser.add_argument('--intermediate',type=int,required=True)
    parser.add_argument('--layers',type=int,required=True)
    parser.add_argument('--parameters',type=int,required=True)
    args=parser.parse_args()
    print(json.dumps(estimate(args.hidden,args.intermediate,args.layers,args.parameters),indent=2))


if __name__=='__main__':main()
