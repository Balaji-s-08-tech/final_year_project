"""Admission for the implemented deterministic, single-continuation evaluation protocol."""
import math


def validate_generation(settings):
    from transformers import GenerationConfig
    if not isinstance(settings,dict) or not settings:raise ValueError('Missing generation configuration')
    unknown=set(settings)-set(GenerationConfig().to_dict())
    if unknown:raise ValueError(f'Unsupported generation fields: {sorted(unknown)}')
    if settings.get('do_sample') is not False or type(settings.get('num_beams'))is not int or settings['num_beams']!=1:
        raise ValueError('Only greedy single-beam evaluation is supported')
    if settings.get('use_cache') is not True:raise ValueError('Controlled evaluation requires KV cache')
    if type(settings.get('max_new_tokens'))is not int or settings['max_new_tokens']<1:raise ValueError('Invalid generation max_new_tokens')
    for name in ('pad_token_id','eos_token_id'):
        value=settings.get(name)
        valid=value is None or (type(value)is int and value>=0) or (name=='eos_token_id' and isinstance(value,list) and value and all(type(v)is int and v>=0 for v in value))
        if name not in settings or not valid:raise ValueError(f'Invalid generation {name}')
    for name in ('num_return_sequences','num_beam_groups'):
        value=settings.get(name,1)
        if type(value)is not int or value!=1:raise ValueError(f'Unsupported generation {name}')
    for name in ('temperature','repetition_penalty','encoder_repetition_penalty'):
        value=settings.get(name,1.)
        if type(value)not in (float,int) or not math.isfinite(value) or value<=0:raise ValueError(f'Invalid generation {name}')
    for name in ('top_p','typical_p'):
        value=settings.get(name,1.)
        if type(value)not in (float,int) or not math.isfinite(value) or not 0<value<=1:raise ValueError(f'Invalid generation {name}')
    if 'top_k' in settings and (type(settings['top_k'])is not int or settings['top_k']<0):raise ValueError('Invalid generation top_k')
    GenerationConfig.from_dict(settings).validate()
    return settings
