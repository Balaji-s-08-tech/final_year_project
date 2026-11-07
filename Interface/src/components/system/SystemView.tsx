import React, { useEffect, useState } from 'react';
import { apiService, SystemInfo } from '../../services/api';

export const SystemView: React.FC = () => {
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    apiService.system().then(setSystem).catch(reason => setError(reason instanceof Error ? reason.message : 'System endpoint unavailable.'));
  }, []);

  const rows: [string, string][] = system ? [
    ['Python', system.python_version],
    ['PyTorch', system.torch_version || 'Unavailable'],
    ['CUDA available', system.cuda.available ? 'Yes' : 'No'],
    ['GPU', system.cuda.name || 'Unavailable'],
    ['Dedicated VRAM total', formatBytes(system.cuda.memory_total_bytes)],
    ['PyTorch allocated VRAM', formatBytes(system.cuda.memory_allocated_bytes)],
    ['System RAM total', formatBytes(system.system_ram.total_bytes)],
    ['System RAM used', formatBytes(system.system_ram.used_bytes)]
  ] : [];

  return <section className="space-y-4"><div><h1 className="text-lg font-bold">System and hardware</h1><p className="text-xs text-neutral-400">Values come from the live backend; unavailable measurements are labeled.</p></div>{error && <div role="alert" className="text-sm text-rose-300">{error}</div>}<div className="rounded-xl border border-neutral-800 divide-y divide-neutral-800">{rows.map(([key, value]) => <div key={key} className="p-3 flex justify-between text-sm"><span className="text-neutral-400">{key}</span><span>{value}</span></div>)}</div></section>;
};

function formatBytes(value: number | null) {
  return value == null ? 'NOT MEASURED' : `${(value / 1024 ** 3).toFixed(2)} GB`;
}
