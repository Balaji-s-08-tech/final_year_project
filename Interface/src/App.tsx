import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { ResearchDemo } from './components/demo/ResearchDemo';
import { Activity, AlertTriangle, CheckCircle2, Cpu, HardDrive, Play, RefreshCw, Square, Terminal } from 'lucide-react';
import { Sidebar, NavView } from './components/navigation/Sidebar';
import { TerminalViewer } from './components/common/TerminalViewer';
import { apiService, BackendHealth, BenchmarkReport, CalibrationFile, Checkpoint, Job, ModelRecord, SystemInfo } from './services/api';

const byteSize = (value?: number | null) => value == null ? 'NOT MEASURED' : `${(value / 1024 ** 3).toFixed(2)} GB`;
const percent = (value?: number | null, total?: number | null) => value == null || total == null || total === 0 ? 'NOT MEASURED' : `${(value / total * 100).toFixed(1)}%`;

export default function App() {
  const [demo, setDemo] = useState(window.location.hash !== '#console');
  return demo ? <ResearchDemo onConsole={() => {window.location.hash='console';setDemo(false);}} /> : <><button className="fixed bottom-3 right-3 z-50 bg-cyan-900 text-white rounded p-2" onClick={() => {window.location.hash='demo';setDemo(true);}}>Research demo</button><ResearchConsole /></>;
}

function ResearchConsole() {
  const [view, setView] = useState<NavView>('dashboard');
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [models, setModels] = useState<ModelRecord[]>([]);
  const [checkpoints, setCheckpoints] = useState<Checkpoint[]>([]);
  const [calibrationFiles, setCalibrationFiles] = useState<CalibrationFile[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);
  const [benchmarks, setBenchmarks] = useState<BenchmarkReport | null>(null);
  const [resultsFile, setResultsFile] = useState<File | null>(null);
  const [resultsImportMessage, setResultsImportMessage] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [operation, setOperation] = useState<'quantize' | 'benchmark' | 'inference'>('quantize');
  const [checkpointId, setCheckpointId] = useState('');
  const [trustedIdentity, setTrustedIdentity] = useState<string | null>(null);
  const selectedCheckpointHash = checkpoints.find(item => item.id === checkpointId)?.checkpoint_sha256 ?? null;
  const selectedIdentity = selectedCheckpointHash ? `${checkpointId}:${selectedCheckpointHash}` : null;
  const trustedCheckpoint = trustedIdentity !== null && trustedIdentity === selectedIdentity;
  const setTrustedCheckpoint = (checked: boolean) => setTrustedIdentity(checked ? selectedIdentity : null);
  useEffect(() => setTrustedIdentity(null), [checkpointId, selectedCheckpointHash]);
  const [calibrationId, setCalibrationId] = useState('');
  const [bits, setBits] = useState<4 | 8>(4);
  const [groupSize, setGroupSize] = useState<32 | 64 | 128 | 256>(128);
  const [prompt, setPrompt] = useState('');
  const [maxNewTokens, setMaxNewTokens] = useState(64);
  const [isTerminalCollapsed, setIsTerminalCollapsed] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [healthResult, systemResult, modelResult, checkpointResult, calibrationResult, jobsResult, benchmarkResult] = await Promise.all([
        apiService.health(), apiService.system(), apiService.models(), apiService.checkpoints(), apiService.calibrationFiles(), apiService.jobs(), apiService.benchmarks()
      ]);
      setHealth(healthResult);
      setSystem(systemResult);
      setModels(modelResult);
      setCheckpoints(checkpointResult);
      setCalibrationFiles(calibrationResult);
      setJobs(jobsResult);
      setBenchmarks(benchmarkResult);
      setError('');
      setCheckpointId(current => current || checkpointResult[0]?.id || '');
      setCalibrationId(current => current || calibrationResult[0]?.id || '');
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Could not reach the backend.');
      setHealth(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    if (!selectedJob || !['queued', 'running'].includes(selectedJob.status)) return;
    const timer = window.setInterval(async () => {
      try {
        const current = await apiService.job(selectedJob.id);
        setSelectedJob(current);
        setJobs(previous => previous.map(item => item.id === current.id ? current : item));
        if (!['queued', 'running'].includes(current.status)) void refresh();
      } catch (reason) {
        setError(reason instanceof Error ? reason.message : 'Job polling failed.');
      }
    }, 1500);
    return () => window.clearInterval(timer);
  }, [selectedJob?.id, selectedJob?.status, refresh]);

  const activeJob = useMemo(() => jobs.find(job => ['queued', 'running'].includes(job.status)) || null, [jobs]);
  const terminalLogs = selectedJob?.logs || activeJob?.logs || [];
  const backendUnavailable = health === null;

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setError('');
    try {
      const payload: Record<string, unknown> = { operation, checkpoint_id: checkpointId, trusted_checkpoint: trustedCheckpoint, checkpoint_sha256: selectedCheckpointHash };
      if (operation === 'quantize') Object.assign(payload, { calibration_id: calibrationId, bits, group_size: groupSize, device: 'cpu' });
      if (operation === 'inference') Object.assign(payload, { prompt, max_new_tokens: maxNewTokens });
      const accepted = await apiService.submitJob(payload);
      const job = await apiService.job(accepted.job_id);
      setSelectedJob(job);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Job submission failed.');
    } finally {
      setSubmitting(false);
    }
  };

  const cancel = async (id: string) => {
    try {
      const job = await apiService.cancelJob(id);
      setSelectedJob(job);
      await refresh();
    } catch (reason) {
      setError(reason instanceof Error ? reason.message : 'Cancellation failed.');
    }
  };

  const importResults = async () => {
    if (!resultsFile) return;
    setResultsImportMessage('');
    try {
      if (resultsFile.size > 2 * 1024 * 1024) throw new Error("Results file exceeds the API 2 MiB request budget.");
      const manifest: unknown = JSON.parse(await resultsFile.text());
      if (!manifest || typeof manifest !== 'object' || Array.isArray(manifest)) throw new Error('Select a JSON object containing benchmark results.');
      await apiService.importResults(manifest as Record<string, unknown>, resultsFile.name);
      setResultsImportMessage('Results manifest imported. Metrics are shown exactly as supplied.');
      setResultsFile(null);
      await refresh();
    } catch (reason) {
      setResultsImportMessage(reason instanceof Error ? reason.message : 'Could not import this results file.');
    }
  };

  const title: Record<NavView, string> = {
    dashboard: 'Research dashboard', experiments: 'Jobs', pipeline: 'Pipeline jobs', models: 'Checkpoints', quantization: 'AWQ quantization', benchmarks: 'Benchmarks', results: 'Measured results', system: 'System and hardware', notes: 'Not connected', presentation: 'Not connected'
  };

  return (
    <div className="flex h-screen w-full bg-neutral-950 text-neutral-100 overflow-hidden font-sans">
      <Sidebar currentView={view} onSelectView={setView} isBackendUnavailable={backendUnavailable} />
      <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
        <header className="border-b border-neutral-800 bg-neutral-950/95 px-5 py-3 flex items-center justify-between gap-3">
          <div>
            <div className="text-sm font-bold">LVLM Compression Lab <span className="text-cyan-300 font-mono">Bunny-v1.0-3B / Phi-2</span></div>
            <div className="text-[11px] text-neutral-500">{title[view]} · API {apiService.baseUrl}</div>
          </div>
          <div className="flex items-center gap-3 text-xs font-mono">
            <span className={health ? 'text-emerald-400' : 'text-rose-400'}>{health?.status === 'ok' ? 'BACKEND CONNECTED' : 'BACKEND UNAVAILABLE'}</span>
            <button onClick={() => void refresh()} className="p-2 rounded border border-neutral-700 text-neutral-300" aria-label="Refresh backend data"><RefreshCw className="w-3.5 h-3.5" /></button>
          </div>
        </header>

        <main className="flex-1 overflow-y-auto p-4 sm:p-6">
          <div className="max-w-7xl mx-auto space-y-5">
            {error && <div role="alert" className="p-3 rounded-lg border border-rose-700/60 bg-rose-950/30 text-sm text-rose-200">{error}</div>}
            {loading && <div className="text-sm text-neutral-400">Connecting to the local API…</div>}

            {view === 'dashboard' && <section className="space-y-5">
              <div className="p-5 rounded-2xl border border-neutral-800 bg-gradient-to-r from-neutral-900 to-neutral-950">
                <div className="text-xs font-mono text-cyan-400 uppercase">Structured pruning and activation-aware quantization</div>
                <h1 className="text-xl font-bold mt-2">Current repository state</h1>
                <p className="text-sm text-neutral-400 mt-2">Dashboard values below are read from this backend. The dashboard does not load Bunny or start ML work.</p>
              </div>
              <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
                <Metric title="API" value={health?.status || 'UNAVAILABLE'} icon={<Activity />} />
                <Metric title="CUDA" value={system?.cuda.available ? system.cuda.name || 'Available' : 'Unavailable'} icon={<Cpu />} />
                <Metric title="Registered checkpoints" value={String(checkpoints.length)} icon={<HardDrive />} />
                <Metric title="Active ML jobs" value={activeJob?.status || 'Idle'} icon={<Activity />} />
              </div>
              {activeJob && <JobCard job={activeJob} onSelect={setSelectedJob} onCancel={cancel} />}
              <section className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/40">
                <h2 className="font-semibold mb-3">Available operations</h2>
                <div className="grid md:grid-cols-3 gap-3 text-sm">
                  <Capability title="AWQ INT4 / INT8" detail="CPU backed; requires a registered structured checkpoint and local calibration text." active />
                  <Capability title="Artifact benchmark" detail="Measures checkpoint size. Latency, quality, and GPU memory remain NOT MEASURED." active />
                  <Capability title="Recovery training" detail="Unavailable: no recovery worker is configured. Cloud jobs are not launched." />
                </div>
              </section>
            </section>}

            {(view === 'pipeline' || view === 'experiments') && <section className="grid xl:grid-cols-[minmax(0,1fr)_minmax(320px,0.8fr)] gap-5">
              <form onSubmit={submit} className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/40 space-y-4">
                <div><h1 className="text-lg font-bold">Submit a supported job</h1><p className="text-xs text-neutral-400 mt-1">Jobs are serialized; ML execution starts only after submission.</p></div>
                <label className="block text-xs text-neutral-300">Operation<select value={operation} onChange={event => setOperation(event.target.value as typeof operation)} className="mt-1 block w-full rounded bg-neutral-950 border border-neutral-700 p-2"><option value="quantize">AWQ quantization</option><option value="benchmark">Artifact size benchmark</option><option value="inference">Text generation from portable AWQ checkpoint</option></select></label>
                <label className="block text-xs text-neutral-300">Checkpoint<select required value={checkpointId} onChange={event => setCheckpointId(event.target.value)} className="mt-1 block w-full rounded bg-neutral-950 border border-neutral-700 p-2"><option value="">Select a registered checkpoint</option>{checkpoints.map(item => <option key={item.id} value={item.id}>{item.name} · {item.precision} · {byteSize(item.size_bytes)}</option>)}</select></label>
                {operation === 'quantize' && <>
                  <label className="block text-xs text-neutral-300">Local calibration text<select required value={calibrationId} onChange={event => setCalibrationId(event.target.value)} className="mt-1 block w-full rounded bg-neutral-950 border border-neutral-700 p-2"><option value="">Select a local text file</option>{calibrationFiles.map(item => <option key={item.id} value={item.id}>{item.name} · {item.size_bytes} bytes</option>)}</select></label>
                  <div className="grid sm:grid-cols-3 gap-3"><label className="text-xs text-neutral-300">Weight bits<select value={bits} onChange={event => setBits(Number(event.target.value) as 4 | 8)} className="mt-1 block w-full rounded bg-neutral-950 border border-neutral-700 p-2"><option value={4}>INT4</option><option value={8}>INT8 control</option></select></label><label className="text-xs text-neutral-300">Group size<select value={groupSize} onChange={event => setGroupSize(Number(event.target.value) as typeof groupSize)} className="mt-1 block w-full rounded bg-neutral-950 border border-neutral-700 p-2">{[32, 64, 128, 256].map(size => <option key={size} value={size}>{size}</option>)}</select></label><div className="text-xs text-neutral-400 flex items-end pb-2">Execution device: CPU</div></div>
                </>}
                {operation === 'inference' && <><label className="block text-xs text-neutral-300">Text prompt<textarea required maxLength={4096} value={prompt} onChange={event => setPrompt(event.target.value)} className="mt-1 block w-full rounded bg-neutral-950 border border-neutral-700 p-2" rows={3} /></label><label className="block text-xs text-neutral-300">Maximum new tokens<input type="number" min={1} max={128} value={maxNewTokens} onChange={event => setMaxNewTokens(Number(event.target.value))} className="mt-1 block w-full rounded bg-neutral-950 border border-neutral-700 p-2" /></label><div className="text-xs text-amber-300">Text only. Image inference is unsupported by the language-only AWQ checkpoint.</div></>}
                {operation === 'benchmark' && <div className="text-xs text-amber-300">This operation measures artifact size only. Latency, throughput, memory, and task accuracy will report NOT MEASURED.</div>}
                {activeJob && <div className="text-xs text-amber-300">Another ML job is {activeJob.status}; wait for it to finish before submitting.</div>}
                {operation !== 'benchmark' && <label className="block text-xs text-amber-200"><input type="checkbox" checked={trustedCheckpoint} onChange={event => setTrustedCheckpoint(event.target.checked)} /> I verified this checkpoint’s source and trust it to execute local Python code.</label>}
                <button disabled={submitting || !health || Boolean(activeJob) || !checkpointId || (operation !== 'benchmark' && (!trustedCheckpoint || !selectedCheckpointHash)) || (operation === 'quantize' && !calibrationId)} className="inline-flex items-center gap-2 px-4 py-2 rounded-lg bg-cyan-500 text-neutral-950 font-semibold text-sm disabled:opacity-40"><Play className="w-4 h-4" />{submitting ? 'Submitting…' : 'Submit job'}</button>
              </form>
              <div className="space-y-4"><section className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/40"><h2 className="font-semibold mb-3">Recent jobs</h2>{jobs.length === 0 ? <Empty>No jobs have been submitted.</Empty> : <div className="space-y-2">{jobs.slice(0, 12).map(job => <JobCard key={job.id} job={job} onSelect={setSelectedJob} onCancel={cancel} />)}</div>}</section>
                {selectedJob && <section className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/40"><h2 className="font-semibold">Selected job · {selectedJob.operation}</h2><p className="text-xs text-neutral-400 mt-1">{selectedJob.id}</p><div className="mt-3"><Progress job={selectedJob} /></div>{selectedJob.error && <p className="text-sm text-rose-300 mt-3">{selectedJob.error}</p>}{selectedJob.result && <pre className="mt-3 text-xs whitespace-pre-wrap break-all text-neutral-300">{JSON.stringify(selectedJob.result, null, 2)}</pre>}</section>}</div>
            </section>}

            {view === 'models' && <section className="space-y-4"><div><h1 className="text-lg font-bold">Registered checkpoints</h1><p className="text-xs text-neutral-400">Only files under configured project checkpoint directories are listed.</p></div>{checkpoints.length === 0 ? <Empty>No checkpoint files are registered.</Empty> : <div className="overflow-auto rounded-xl border border-neutral-800"><table className="w-full text-sm"><thead className="bg-neutral-900 text-neutral-400"><tr><th className="p-3 text-left">File</th><th className="p-3 text-left">Format / precision</th><th className="p-3 text-right">Size</th><th className="p-3 text-left">Lineage</th><th className="p-3 text-left">Modality</th></tr></thead><tbody>{checkpoints.map(item => <tr key={item.id} className="border-t border-neutral-800"><td className="p-3"><div>{item.name}</div><div className="text-[10px] text-neutral-500 break-all">{item.path}</div></td><td className="p-3">{item.format} · {item.precision}</td><td className="p-3 text-right font-mono">{byteSize(item.size_bytes)}</td><td className="p-3 text-neutral-400">{item.lineage}</td><td className="p-3">{item.multimodal ? 'Vision + text' : 'Language only / unknown'}</td></tr>)}</tbody></table></div>}{models.map(model => <div key={model.id} className="text-xs text-neutral-500">{model.name} · {model.backbone} · architecture import {model.architecture_importable ? 'available' : 'failed'}</div>)}</section>}

            {view === 'quantization' && <section className="max-w-3xl p-5 rounded-2xl border border-neutral-800 bg-neutral-900/40 space-y-3"><h1 className="text-lg font-bold">Bunny Phi-2 weight-only quantization</h1><p className="text-sm text-neutral-300">The project integration calibrates the selected checkpoint, searches equivalent scales, clips weights, then writes a portable packed weight checkpoint.</p><p className="text-xs text-amber-300">The current portable linear layer dequantizes weights during PyTorch forward execution; it does not provide an optimized AWQ CUDA kernel or imply lower latency.</p><p className="text-xs text-neutral-400">Calibration comes from a registered local text file. The service uses four samples and sequence length 32 for UI-submitted jobs.</p><button onClick={() => setView('pipeline')} className="px-3 py-2 rounded bg-cyan-500 text-neutral-950 text-sm font-semibold">Configure quantization job</button></section>}

            {view === 'benchmarks' && <section className="space-y-4"><div><h1 className="text-lg font-bold">Benchmark records</h1><p className="text-xs text-neutral-400">Imported values retain their source; missing metrics remain NOT MEASURED.</p></div><div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/40 space-y-3"><label className="block text-xs text-neutral-300">Import a cloud benchmark manifest (.json)<input type="file" accept="application/json,.json" onChange={event => setResultsFile(event.target.files?.[0] || null)} className="mt-2 block w-full text-xs text-neutral-400" /></label><button onClick={() => void importResults()} disabled={!resultsFile} className="px-3 py-2 rounded bg-cyan-500 text-neutral-950 text-sm font-semibold disabled:opacity-40">Import results</button>{resultsImportMessage && <p role="status" className="text-xs text-neutral-300">{resultsImportMessage}</p>}</div>{!benchmarks?.items.length ? <Empty>No benchmark manifest is available. Accuracy, latency, tokens/sec, and peak memory are NOT MEASURED.</Empty> : benchmarks.items.map(item => <section key={item.path} className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/40"><div className="mb-2 text-xs text-cyan-300">{item.origin}{item.source_label ? ` · ${item.source_label}` : ''}{item.imported_at ? ` · imported ${item.imported_at}` : ''}</div>{Object.keys(item.metadata || {}).length > 0 && <pre className="mb-3 text-xs whitespace-pre-wrap break-all text-neutral-400">{JSON.stringify(item.metadata, null, 2)}</pre>}<pre className="text-xs overflow-auto">{JSON.stringify(item.results, null, 2)}</pre><div className="text-neutral-500 mt-2 text-[10px]">{item.path}</div></section>)}<button onClick={() => setView('pipeline')} className="px-3 py-2 rounded border border-neutral-700 text-sm">Submit artifact size benchmark</button></section>}

            {view === 'results' && <section className="space-y-4"><h1 className="text-lg font-bold">Measured checkpoint information</h1><p className="text-xs text-neutral-400">File sizes are filesystem measurements. Versioned records retain their measurement status, unit, method and source; null values are not zero.</p>{checkpoints.map(item => <div key={item.id} className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/40 flex justify-between gap-4"><div><div className="font-medium">{item.name}</div><div className="text-xs text-neutral-500">{item.precision} · {item.lineage}</div></div><div className="font-mono text-cyan-300">{byteSize(item.size_bytes)}</div></div>)}{benchmarks?.items.map(item => <div key={item.path} className="p-4 rounded-xl border border-neutral-800"><p className="text-xs text-cyan-300">{item.origin} · {item.source_label || item.path}</p><pre className="text-xs overflow-auto">{JSON.stringify(item.results, null, 2)}</pre></div>)}</section>}

            {view === 'system' && <section className="space-y-4"><h1 className="text-lg font-bold">Live system information</h1>{!system ? <Empty>System information unavailable while the backend is disconnected.</Empty> : <div className="grid md:grid-cols-2 gap-4"><InfoCard title="Runtime" rows={[["Python", system.python_version], ["PyTorch", system.torch_version || 'Unavailable'], ["CUDA runtime", system.cuda.version || 'Unavailable'], ["CUDA available", system.cuda.available ? 'Yes' : 'No']]} /><InfoCard title="Dedicated GPU memory" rows={[["Device", system.cuda.name || 'Unavailable'], ["Total", byteSize(system.cuda.memory_total_bytes)], ["Allocated by PyTorch", byteSize(system.cuda.memory_allocated_bytes)], ["Reserved by PyTorch", byteSize(system.cuda.memory_reserved_bytes)], ["Free", byteSize(system.cuda.memory_free_bytes)]]} /><InfoCard title="System RAM" rows={[["Total", byteSize(system.system_ram.total_bytes)], ["Used", byteSize(system.system_ram.used_bytes)], ["Available", byteSize(system.system_ram.available_bytes)], ["Usage", `${system.system_ram.percent_used}%`]]} /><InfoCard title="Backend" rows={[["Service", health?.service || 'Unavailable'], ["Version", health?.version || 'Unavailable'], ["API URL", apiService.baseUrl]]} /></div>}</section>}

            {!['dashboard', 'pipeline', 'experiments', 'models', 'quantization', 'benchmarks', 'results', 'system'].includes(view) && <Empty>This view is not connected to measured backend data yet.</Empty>}
          </div>
        </main>

        <div className="border-t border-neutral-800 bg-neutral-950">
          <button onClick={() => setIsTerminalCollapsed(value => !value)} className="w-full px-4 py-2 flex items-center gap-2 text-xs text-neutral-300"><Terminal className="w-3.5 h-3.5 text-cyan-400" />Job logs {selectedJob ? `· ${selectedJob.id.slice(0, 8)}` : ''}<span className="ml-auto">{isTerminalCollapsed ? 'Expand' : 'Collapse'}</span></button>
          {!isTerminalCollapsed && <div className="px-3 pb-3"><TerminalViewer logs={terminalLogs.map((entry, index) => ({ id: `${selectedJob?.id || 'job'}-${index}`, timestamp: new Date(entry.timestamp).toLocaleTimeString(), level: entry.level as 'INFO' | 'WARNING' | 'ERROR' | 'SUCCESS', stage: selectedJob?.operation || activeJob?.operation || 'Backend', message: entry.message }))} onClear={() => setSelectedJob(null)} maxHeight="145px" /></div>}
        </div>
      </div>
    </div>
  );
}

function Metric({ title, value, icon }: { title: string; value: string; icon: React.ReactNode }) {
  return <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/50"><div className="flex items-center justify-between text-xs text-neutral-400">{title}<span className="text-cyan-400">{icon}</span></div><div className="mt-2 text-sm font-semibold break-words">{value}</div></div>;
}

function Capability({ title, detail, active = false }: { title: string; detail: string; active?: boolean }) {
  return <div className="p-3 rounded-lg border border-neutral-800 bg-neutral-950/60"><div className="flex gap-2 items-center font-medium"><span className={active ? 'text-emerald-400' : 'text-amber-400'}>{active ? <CheckCircle2 className="w-4 h-4" /> : <AlertTriangle className="w-4 h-4" />}</span>{title}</div><p className="text-xs text-neutral-400 mt-2">{detail}</p></div>;
}

function Empty({ children }: { children: React.ReactNode }) {
  return <div className="p-5 rounded-xl border border-dashed border-neutral-700 text-sm text-neutral-400">{children}</div>;
}

function Progress({ job }: { job: Job }) {
  const measured = job.progress != null;
  const width = measured ? `${job.progress}%` : job.status === 'running' ? '35%' : '0%';
  return <div><div className="flex justify-between text-xs mb-1"><span>{job.status}</span><span>{measured ? `${job.progress}%` : job.status === 'running' ? 'Working' : 'Queued'}</span></div><div className="h-1.5 rounded bg-neutral-800"><div className={`h-full bg-cyan-400 rounded ${!measured && job.status === 'running' ? 'animate-pulse' : ''}`} style={{ width }} /></div></div>;
}

function JobCard({ job, onSelect, onCancel }: { key?: React.Key; job: Job; onSelect: (job: Job) => void; onCancel: (id: string) => void }) {
  return <div className="p-3 rounded-lg border border-neutral-800 bg-neutral-950/50"><div className="flex items-center justify-between gap-2"><button onClick={() => onSelect(job)} className="text-left min-w-0"><div className="text-sm font-medium">{job.operation} · {job.status}</div><div className="text-[10px] text-neutral-500 font-mono truncate">{job.id}</div></button>{['queued', 'running'].includes(job.status) && <button onClick={() => onCancel(job.id)} aria-label="Cancel job" className="p-2 text-rose-300"><Square className="w-3.5 h-3.5" /></button>}</div><div className="mt-2"><Progress job={job} /></div>{job.error && <div className="text-xs text-rose-300 mt-2">{job.error}</div>}</div>;
}

function InfoCard({ title, rows }: { title: string; rows: [string, string][] }) {
  return <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/40"><h2 className="font-semibold mb-3">{title}</h2><dl className="space-y-2">{rows.map(([label, value]) => <div key={label} className="flex justify-between gap-3 text-xs"><dt className="text-neutral-500">{label}</dt><dd className="text-neutral-200 text-right break-all">{value}</dd></div>)}</dl></div>;
}
