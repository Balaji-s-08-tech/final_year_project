"""One bounded JSON number contract: signed INT64 and finite binary64, no underflow."""
import json
import math
from decimal import Decimal

MIN_INT=-(2**63)
MAX_INT=2**63-1


def validate(value, *, max_depth=64):
    """Validate decoded or in-memory JSON recursively, including metric metadata."""
    def visit(item,depth,ancestors):
        if depth>max_depth:raise ValueError('JSON nesting exceeds supported depth')
        if item is None or type(item) in (str,bool):return
        if type(item)is int:
            if not MIN_INT<=item<=MAX_INT:raise ValueError('JSON integer outside signed INT64 range')
            return
        if type(item)is float:
            if not math.isfinite(item):raise ValueError('Nonfinite JSON number')
            return
        if isinstance(item,(dict,list,tuple)):
            if id(item) in ancestors:raise ValueError('Circular JSON structure')
            chain=ancestors|{id(item)}
            if isinstance(item,dict):
                keys=set()
                for key,child in item.items():
                    if type(key) not in (str,int):raise ValueError('Invalid JSON object key')
                    normalized=str(key)
                    if normalized in keys:raise ValueError('Ambiguous JSON object keys')
                    keys.add(normalized);visit(child,depth+1,chain)
            else:
                for child in item:visit(child,depth+1,chain)
            return
        raise ValueError(f'Unsupported JSON value type: {type(item).__name__}')
    visit(value,0,set())
    return value


def _integer(text):return validate(int(text))


def _float(text):
    value=float(text)
    validate(value)
    if value==0 and not Decimal(text).is_zero():raise ValueError('JSON number underflows binary64')
    return value


def _constant(text):raise ValueError(f'Nonfinite JSON number: {text}')


def _object(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError(f'Duplicate JSON key: {key}')
        result[key]=value
    return result


def loads(text):
    try:return validate(json.loads(text,parse_int=_integer,parse_float=_float,parse_constant=_constant,object_pairs_hook=_object))
    except RecursionError as exc:raise ValueError('JSON nesting exceeds supported depth') from exc
