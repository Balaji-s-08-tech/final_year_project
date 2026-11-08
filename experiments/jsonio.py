"""One bounded JSON number contract: signed INT64 and finite binary64, no underflow."""
import json
import math
from decimal import Decimal
from dataclasses import dataclass
from pathlib import Path
import os
import stat


@dataclass(frozen=True)
class JsonLimits:
    max_bytes: int = 8*1024*1024
    max_nodes: int = 100000
    max_string_bytes: int = 1024*1024
    max_key_bytes: int = 4096
    max_depth: int = 64
    max_number_chars: int = 128

    def __post_init__(self):
        if any(type(v)is not int or v<1 for v in self.__dict__.values()):raise ValueError("JSON limits must be positive integers")


DEFAULT_LIMITS=JsonLimits()


def read_text(path, *, max_bytes=DEFAULT_LIMITS.max_bytes):
    if type(max_bytes)is not int or max_bytes<1:raise ValueError("Positive byte budget required")
    fd=os.open(Path(path),os.O_RDONLY|os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):raise ValueError("Only regular input files are supported")
        if info.st_size>max_bytes:raise ValueError("Input exceeds byte budget")
        data=stream.read(max_bytes+1)
        if len(data)>max_bytes:raise ValueError("Input exceeds byte budget")
    return data.decode("utf-8",errors="strict")


def read(path, *, limits=DEFAULT_LIMITS):
    return loads(read_text(path,max_bytes=limits.max_bytes),limits=limits)


def _preflight(text,limits):
    depth=nodes=0;quoted=escaped=primitive=False;size=0;number_size=0
    for char in text:
        if quoted:
            size+=1
            if size>limits.max_string_bytes:raise ValueError("JSON string exceeds limit")
            if escaped:escaped=False
            elif char=="\\":escaped=True
            elif char=='"':quoted=False
            continue
        if char=='"':quoted=True;size=0;nodes+=1;primitive=False
        elif char in "[{":depth+=1;nodes+=1;primitive=False
        elif char in "]}":depth-=1;primitive=False
        elif char in ",:" or char.isspace():primitive=False
        else:
            if not primitive:nodes+=1;number_size=0
            primitive=True;number_size+=1
            if number_size>limits.max_number_chars:raise ValueError("JSON primitive exceeds token limit")
        if depth>limits.max_depth:raise ValueError("JSON nesting exceeds supported depth")
        if nodes>limits.max_nodes:raise ValueError("JSON node count exceeds limit")


MIN_INT=-(2**63)
MAX_INT=2**63-1


def validate(value, *, max_depth=None, limits=DEFAULT_LIMITS):
    """Bound an in-memory graph before serialization; keys count as nodes."""
    depth_limit=limits.max_depth if max_depth is None else max_depth
    if type(depth_limit)is not int or depth_limit<1:raise ValueError('Positive depth limit required')
    nodes=total=0
    def string(text,key=False):
        nonlocal total
        bound=limits.max_key_bytes if key else limits.max_string_bytes
        if len(text)>bound:raise ValueError('JSON key/string exceeds limit')
        size=len(text.encode('utf-8',errors='strict'))
        if size>bound:raise ValueError('JSON key/string exceeds byte limit')
        total+=size
    def visit(item,depth,ancestors):
        nonlocal nodes,total
        nodes+=1;total+=1
        if nodes>limits.max_nodes or total>limits.max_bytes:raise ValueError('JSON graph exceeds node/byte budget')
        if depth>depth_limit:raise ValueError('JSON nesting exceeds supported depth')
        if type(item)is str:string(item)
        elif item is None or type(item)is bool:total+=5
        elif type(item)is int:
            if not MIN_INT<=item<=MAX_INT:raise ValueError('JSON integer outside signed INT64 range')
            total+=20
        elif type(item)is float:
            if not math.isfinite(item):raise ValueError('Nonfinite JSON number')
            total+=24
        elif isinstance(item,(dict,list,tuple)):
            if id(item) in ancestors:raise ValueError('Circular JSON structure')
            chain=ancestors|{id(item)}
            if isinstance(item,dict):
                keys=set()
                for key,child in item.items():
                    if type(key)not in (str,int):raise ValueError('Invalid JSON object key')
                    normalized=str(key)
                    if normalized in keys:raise ValueError('Ambiguous JSON object keys')
                    keys.add(normalized);nodes+=1;string(normalized,True);visit(child,depth+1,chain)
            else:
                for child in item:visit(child,depth+1,chain)
        else:raise ValueError(f'Unsupported JSON value type: {type(item).__name__}')
        if total>limits.max_bytes:raise ValueError('JSON graph exceeds byte budget')
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


def loads(text, *, limits=DEFAULT_LIMITS):
    if not isinstance(text,(str,bytes,bytearray)):raise ValueError('JSON source must be text/bytes')
    if len(text)>limits.max_bytes:raise ValueError('JSON exceeds byte budget')
    if not isinstance(text,str):text=bytes(text).decode('utf-8',errors='strict')
    if len(text.encode('utf-8',errors='strict'))>limits.max_bytes:raise ValueError('JSON exceeds byte budget')
    _preflight(text,limits)
    try:return validate(json.loads(text,parse_int=_integer,parse_float=_float,parse_constant=_constant,object_pairs_hook=_object),limits=limits)
    except RecursionError as exc:raise ValueError('JSON nesting exceeds supported depth') from exc


def lines(path, *, max_bytes=64*1024*1024, max_line_bytes=1024*1024, max_records=100000):
    for value in (max_bytes,max_line_bytes,max_records):
        if type(value)is not int or value<1:raise ValueError('Positive JSONL/text limits required')
    fd=os.open(Path(path),os.O_RDONLY|os.O_NONBLOCK)
    count=total=0
    with os.fdopen(fd,'rb') as stream:
        info=os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode):raise ValueError('Only regular input files are supported')
        if info.st_size>max_bytes:raise ValueError('JSONL/text exceeds byte budget')
        while True:
            line=stream.readline(max_line_bytes+1)
            if not line:break
            count+=1;total+=len(line)
            if count>max_records or total>max_bytes or len(line)>max_line_bytes:raise ValueError('JSONL/text exceeds record/line/byte budget')
            yield line


def files(root, pattern, *, max_entries=10000):
    """Bound discovery as well as returned files; do not follow directory symlinks."""
    from fnmatch import fnmatch
    if type(max_entries) is not int or max_entries < 1:
        raise ValueError('Positive discovery budget required')
    count = 0
    for directory, folders, names in os.walk(root, followlinks=False):
        count += len(folders) + len(names)
        if count > max_entries:
            raise ValueError('File discovery exceeds entry budget')
        folders[:] = [name for name in folders if not name.startswith('.')]
        for name in names:
            if not name.startswith('.') and fnmatch(name, pattern):
                yield Path(directory) / name
