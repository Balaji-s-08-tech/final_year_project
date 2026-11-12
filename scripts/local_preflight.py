"""Report the local runtime without allocating model tensors or a CUDA context."""
import json
import platform
import shutil
import subprocess


def main():
    import psutil
    import torch
    report={'python':platform.python_version(),'platform':platform.platform(),
            'torch':torch.__version__,'torch_cuda_runtime':torch.version.cuda,
            'available_ram_bytes':psutil.virtual_memory().available,
            'available_disk_bytes':shutil.disk_usage('.').free,
            'memory_feasibility':'NOT_PROVEN','gpus':[]}
    if shutil.which('nvidia-smi'):
        result=subprocess.run(['nvidia-smi','--query-gpu=name,memory.total,memory.free','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
        report['nvidia_smi_status']=result.returncode
        if result.returncode==0:
            report['gpus']=[line.strip() for line in result.stdout.splitlines()]
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
