import React, { useState } from 'react';
import {
  ExperimentConfig,
  PipelineStageInfo,
  LogEntry,
  ExecutionEnvironment,
  PruningCriterion,
  QuantizationMethod
} from '../../types/research';
import { TerminalViewer } from '../common/TerminalViewer';
import { ResearchBadge, ExecutionBadge } from '../common/Badge';
import {
  Play,
  Square,
  RotateCcw,
  Download,
  PlusCircle,
  Cpu,
  Layers,
  Settings,
  Sliders,
  Check,
  AlertCircle,
  Clock,
  Sparkles
} from 'lucide-react';

interface ExperimentsViewProps {
  experiments: ExperimentConfig[];
  currentExperiment: ExperimentConfig;
  stages: PipelineStageInfo[];
  logs: LogEntry[];
  isRunning: boolean;
  onRunExperiment: (id: string) => void;
  onStopExperiment: (id: string) => void;
  onResetPipeline: () => void;
  onCreateExperiment: (config: Omit<ExperimentConfig, 'id' | 'createdAt'>) => void;
  onSelectExperiment: (exp: ExperimentConfig) => void;
  onExportBundle: (id: string) => void;
  onClearLogs: () => void;
}

export const ExperimentsView: React.FC<ExperimentsViewProps> = ({
  experiments,
  currentExperiment,
  stages,
  logs,
  isRunning,
  onRunExperiment,
  onStopExperiment,
  onResetPipeline,
  onCreateExperiment,
  onSelectExperiment,
  onExportBundle,
  onClearLogs
}) => {
  const [showConfigModal, setShowConfigModal] = useState<boolean>(false);
  const [pruningRatio, setPruningRatio] = useState<number>(0.20);
  const [pruningCriterion, setPruningCriterion] = useState<PruningCriterion>('Taylor FO');
  const [pruningMode, setPruningMode] = useState<'width-wise' | 'layer-wise'>('width-wise');
  const [recoveryEnabled, setRecoveryEnabled] = useState<boolean>(true);
  const [trainingMethod, setTrainingMethod] = useState<'LoRA' | 'Full Fine-Tuning' | 'Adapter Only'>('LoRA');
  const [epochs, setEpochs] = useState<number>(3);
  const [learningRate, setLearningRate] = useState<number>(0.0002);
  const [loraRank, setLoraRank] = useState<number>(16);
  const [quantMethod, setQuantMethod] = useState<QuantizationMethod>('AWQ INT4');
  const [calibDataset, setCalibDataset] = useState<'ShareGPT4V' | 'Pile-val' | 'COCO-QA' | 'Wikitext-2'>('ShareGPT4V');
  const [calibSamples, setCalibSamples] = useState<number>(128);
  const [groupSize, setGroupSize] = useState<64 | 128>(128);
  const [symmetric, setSymmetric] = useState<boolean>(true);
  const [execTarget, setExecTarget] = useState<ExecutionEnvironment>('CLOUD');
  const [newExpName, setNewExpName] = useState<string>('Custom Pruned + AWQ INT4 Study');

  const runningStage = stages.find(s => s.status === 'running');
  const completedStagesCount = stages.filter(s => s.status === 'completed').length;
  const overallProgress = Math.round((completedStagesCount / stages.length) * 100);

  const handleCreateSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onCreateExperiment({
      name: newExpName,
      model: 'Bunny-v1_0-3B',
      pruning: {
        method: 'LLM-Pruner Structured',
        targetRatio: pruningRatio,
        mode: pruningMode,
        criterion: pruningCriterion,
        targetModules: ['q_proj', 'k_proj', 'v_proj', 'fc1', 'fc2']
      },
      recovery: {
        enabled: recoveryEnabled,
        trainingMethod: trainingMethod,
        epochs: epochs,
        learningRate: learningRate,
        loraRank: loraRank,
        loraAlpha: loraRank * 2,
        batchSize: 4
      },
      quantization: {
        method: quantMethod,
        awq: quantMethod === 'AWQ INT4' ? {
          calibrationDataset: calibDataset,
          calibrationSamples: calibSamples,
          weightBits: 4,
          groupSize: groupSize,
          symmetric: symmetric,
          activationAwareSearch: true,
          dualityRatio: 0.5
        } : undefined
      },
      execution: execTarget
    });
    setShowConfigModal(false);
  };

  return (
    <div id="experiments-view" className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-2xl border border-neutral-800 bg-neutral-900/60">
        <div>
          <div className="flex items-center gap-2">
            <h2 className="text-base font-bold text-neutral-100">Experiment Control Center</h2>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-cyan-950/60 text-cyan-300 border border-cyan-800/60">
              Active: {currentExperiment.id}
            </span>
          </div>
          <p className="text-xs text-neutral-400 mt-0.5">
            Configure pipeline hyperparameters, inspect configuration summaries, and execute reproducible compression runs.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            id="btn-open-create-exp"
            onClick={() => setShowConfigModal(true)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-neutral-700 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-mono transition-colors"
          >
            <PlusCircle className="w-3.5 h-3.5 text-cyan-400" />
            <span>Create Experiment</span>
          </button>

          <button
            id="btn-export-bundle"
            onClick={() => onExportBundle(currentExperiment.id)}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl border border-neutral-700 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 text-xs font-mono transition-colors"
          >
            <Download className="w-3.5 h-3.5" />
            <span>Export Results</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
            <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-cyan-400" />
              <span>Configuration Summary</span>
            </h3>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
              {currentExperiment.id}
            </span>
          </div>

          <div className="p-3.5 rounded-xl border border-neutral-800/80 bg-neutral-950/70 font-mono text-xs space-y-2.5">
            <div className="flex justify-between border-b border-neutral-900 pb-1.5">
              <span className="text-neutral-500">Experiment ID:</span>
              <span className="text-cyan-300 font-semibold">{currentExperiment.id}</span>
            </div>
            <div className="flex justify-between border-b border-neutral-900 pb-1.5">
              <span className="text-neutral-500">Model:</span>
              <span className="text-neutral-200">{currentExperiment.model}</span>
            </div>
            <div className="flex justify-between border-b border-neutral-900 pb-1.5">
              <span className="text-neutral-500">Vision Encoder:</span>
              <span className="text-neutral-200">SigLIP-SO400M (428M)</span>
            </div>
            <div className="flex justify-between border-b border-neutral-900 pb-1.5">
              <span className="text-neutral-500">Language Backbone:</span>
              <span className="text-neutral-200">Phi-2 (2.78B)</span>
            </div>
            <div className="flex justify-between border-b border-neutral-900 pb-1.5">
              <span className="text-neutral-500">Pruning Target:</span>
              <span className="text-neutral-200 font-semibold">
                {Math.round(currentExperiment.pruning.targetRatio * 100)}% ({currentExperiment.pruning.criterion})
              </span>
            </div>
            <div className="flex justify-between border-b border-neutral-900 pb-1.5">
              <span className="text-neutral-500">Recovery Mode:</span>
              <span className={currentExperiment.recovery.enabled ? 'text-emerald-400' : 'text-neutral-400'}>
                {currentExperiment.recovery.enabled
                  ? `${currentExperiment.recovery.trainingMethod} (${currentExperiment.recovery.epochs} epochs)`
                  : 'Disabled'}
              </span>
            </div>
            <div className="flex justify-between border-b border-neutral-900 pb-1.5">
              <span className="text-neutral-500">Quantization:</span>
              <span className="text-cyan-400 font-bold">{currentExperiment.quantization.method}</span>
            </div>
            {currentExperiment.quantization.awq && (
              <div className="flex justify-between border-b border-neutral-900 pb-1.5">
                <span className="text-neutral-500">AWQ Group Size:</span>
                <span className="text-neutral-200">
                  {currentExperiment.quantization.awq.groupSize} (W4A16)
                </span>
              </div>
            )}
            <div className="flex justify-between pt-1">
              <span className="text-neutral-500">Target Worker:</span>
              <span className="text-amber-400 font-bold">{currentExperiment.execution} GPU</span>
            </div>
          </div>

          <div className="pt-2 flex flex-col gap-2">
            {!isRunning ? (
              <button
                id="btn-run-experiment"
                onClick={() => onRunExperiment(currentExperiment.id)}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-neutral-950 font-mono text-xs font-bold transition-all shadow-[0_0_20px_rgba(6,182,212,0.3)]"
              >
                <Play className="w-4 h-4 fill-current" />
                <span>Run Experiment (Demo Mode)</span>
              </button>
            ) : (
              <button
                id="btn-stop-experiment"
                onClick={() => onStopExperiment(currentExperiment.id)}
                className="w-full flex items-center justify-center gap-2 py-3 rounded-xl bg-rose-500/20 hover:bg-rose-500/30 text-rose-300 border border-rose-500/50 font-mono text-xs font-bold transition-all"
              >
                <Square className="w-4 h-4 fill-current" />
                <span>Stop Experiment</span>
              </button>
            )}

            <button
              id="btn-reset-pipeline"
              onClick={onResetPipeline}
              className="w-full flex items-center justify-center gap-2 py-2 rounded-xl border border-neutral-800 bg-neutral-950 hover:bg-neutral-900 text-neutral-400 hover:text-neutral-200 font-mono text-xs transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Reset Pipeline State</span>
            </button>
          </div>

          <div className="p-3 rounded-xl border border-neutral-800/80 bg-neutral-950/50 text-[11px] text-neutral-400 leading-relaxed">
            <strong className="text-neutral-300">Execution Notice:</strong> Frontend dispatches calls through the modular API layer. When backend REST is detached, pipeline states execute through the simulated runner marked <span className="text-cyan-400 font-mono font-semibold">DEMO MODE</span>.
          </div>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <div>
                <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
                  <Clock className="w-4 h-4 text-cyan-400" />
                  <span>Pipeline Execution Tracker</span>
                </h3>
                <p className="text-xs text-neutral-400 mt-0.5">
                  10-Stage Sequential Compression Pipeline
                </p>
              </div>
              <div className="flex items-center gap-3 font-mono text-xs">
                <span className="text-neutral-400">Stages: {completedStagesCount} / {stages.length}</span>
                <span className="px-2 py-0.5 rounded bg-neutral-800 text-cyan-300 font-bold">
                  {overallProgress}%
                </span>
              </div>
            </div>

            <div className="pt-4 grid grid-cols-2 sm:grid-cols-5 gap-2.5">
              {stages.map(stage => {
                const isStageRunning = stage.status === 'running';
                return (
                  <div
                    key={stage.id}
                    className={`p-2.5 rounded-xl border transition-all text-xs flex flex-col justify-between ${
                      isStageRunning
                        ? 'bg-cyan-950/50 border-cyan-500/60 shadow-[0_0_12px_rgba(6,182,212,0.2)]'
                        : stage.status === 'completed'
                        ? 'bg-neutral-950/70 border-emerald-900/40 text-neutral-300'
                        : stage.status === 'blocked'
                        ? 'bg-neutral-950/70 border-rose-900/40 text-neutral-400'
                        : 'bg-neutral-950/40 border-neutral-800/80 text-neutral-500'
                    }`}
                  >
                    <div>
                      <div className="flex items-center justify-between text-[10px] font-mono mb-1">
                        <span className="text-neutral-500">0{stage.id}</span>
                        <ExecutionBadge status={stage.status} />
                      </div>
                      <div className="font-semibold text-neutral-200 text-[11px] leading-tight line-clamp-1">
                        {stage.name}
                      </div>
                    </div>
                    <div className="mt-2 pt-1 border-t border-neutral-900 text-[10px] font-mono flex items-center justify-between">
                      <span className="text-neutral-500">{stage.runtime || '--'}</span>
                      <ResearchBadge status={stage.researchStatus} size="sm" />
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          <TerminalViewer logs={logs} onClear={onClearLogs} />
        </div>
      </div>

      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
          <div>
            <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Experiment Repository &amp; Run History</span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Comparative study runs with verified vs scheduled configurations.
            </p>
          </div>
          <span className="text-xs font-mono text-neutral-500">{experiments.length} registered runs</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-[11px]">
                <th className="py-2.5 px-3">Experiment ID</th>
                <th className="py-2.5 px-3">Model</th>
                <th className="py-2.5 px-3">Pruning Ratio</th>
                <th className="py-2.5 px-3">Recovery</th>
                <th className="py-2.5 px-3">Quantization</th>
                <th className="py-2.5 px-3">Worker</th>
                <th className="py-2.5 px-3">Created</th>
                <th className="py-2.5 px-3 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-850">
              {experiments.map(exp => {
                const isCurrent = exp.id === currentExperiment.id;
                return (
                  <tr
                    key={exp.id}
                    className={`hover:bg-neutral-800/40 transition-colors ${
                      isCurrent ? 'bg-neutral-800/30' : ''
                    }`}
                  >
                    <td className="py-3 px-3 font-bold text-neutral-200 flex items-center gap-2">
                      {isCurrent && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400" />}
                      <span>{exp.id}</span>
                    </td>
                    <td className="py-3 px-3 text-neutral-300">{exp.model}</td>
                    <td className="py-3 px-3 text-neutral-300">
                      {exp.pruning.targetRatio > 0
                        ? `${Math.round(exp.pruning.targetRatio * 100)}% (${exp.pruning.criterion})`
                        : 'None (Baseline)'}
                    </td>
                    <td className="py-3 px-3">
                      {exp.recovery.enabled ? (
                        <span className="text-emerald-400">LoRA (r={exp.recovery.loraRank})</span>
                      ) : (
                        <span className="text-neutral-500">Disabled</span>
                      )}
                    </td>
                    <td className="py-3 px-3 font-semibold text-cyan-300">{exp.quantization.method}</td>
                    <td className="py-3 px-3 text-neutral-400">{exp.execution}</td>
                    <td className="py-3 px-3 text-neutral-500">{exp.createdAt}</td>
                    <td className="py-3 px-3 text-right">
                      <button
                        onClick={() => onSelectExperiment(exp)}
                        className={`px-2.5 py-1 rounded text-[11px] font-mono transition-colors ${
                          isCurrent
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40'
                            : 'bg-neutral-800 text-neutral-300 hover:bg-neutral-700'
                        }`}
                      >
                        {isCurrent ? 'Active' : 'Load Config'}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {showConfigModal && (
        <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="w-full max-w-2xl bg-neutral-900 border border-neutral-700 rounded-2xl shadow-2xl p-6 space-y-5 max-h-[90vh] overflow-y-auto">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <div>
                <h3 className="text-base font-bold text-neutral-100">Create New VLM Compression Experiment</h3>
                <p className="text-xs text-neutral-400 mt-0.5">Specify pruning dynamics, recovery regime, and AWQ calibration</p>
              </div>
              <button
                onClick={() => setShowConfigModal(false)}
                className="text-neutral-400 hover:text-neutral-200 text-sm font-mono"
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleCreateSubmit} className="space-y-4 text-xs font-mono">
              <div>
                <label className="block text-neutral-300 mb-1">Experiment Name</label>
                <input
                  type="text"
                  value={newExpName}
                  onChange={e => setNewExpName(e.target.value)}
                  className="w-full px-3 py-2 rounded-lg bg-neutral-950 border border-neutral-700 text-neutral-200 focus:outline-none focus:border-cyan-500"
                  required
                />
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950/60 space-y-3">
                  <div className="text-neutral-300 font-bold text-[11px] uppercase tracking-wider text-cyan-400">
                    1. Structured Pruning Configuration
                  </div>
                  <div>
                    <label className="block text-neutral-400 mb-1">Target Pruning Ratio</label>
                    <select
                      value={pruningRatio}
                      onChange={e => setPruningRatio(parseFloat(e.target.value))}
                      className="w-full px-2.5 py-1.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-200"
                    >
                      <option value={0.10}>10% Width Reduction</option>
                      <option value={0.20}>20% Width Reduction (Standard)</option>
                      <option value={0.30}>30% Width Reduction (Aggressive)</option>
                      <option value={0.0}>0% (Unpruned Baseline)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-neutral-400 mb-1">Importance Criterion</label>
                    <select
                      value={pruningCriterion}
                      onChange={e => setPruningCriterion(e.target.value as PruningCriterion)}
                      className="w-full px-2.5 py-1.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-200"
                    >
                      <option value="Taylor FO">Taylor First-Order (FO)</option>
                      <option value="L1 Norm">L1 Weight Magnitude</option>
                      <option value="Hessian">Second-Order Hessian</option>
                      <option value="Wanda">Wanda (Weight &amp; Activation)</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-neutral-400 mb-1">Pruning Mode</label>
                    <div className="flex gap-2">
                      <button
                        type="button"
                        onClick={() => setPruningMode('width-wise')}
                        className={`flex-1 py-1 rounded border text-center ${
                          pruningMode === 'width-wise'
                            ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                            : 'border-neutral-800 text-neutral-400'
                        }`}
                      >
                        Width-Wise
                      </button>
                      <button
                        type="button"
                        onClick={() => setPruningMode('layer-wise')}
                        className={`flex-1 py-1 rounded border text-center ${
                          pruningMode === 'layer-wise'
                            ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                            : 'border-neutral-800 text-neutral-400'
                        }`}
                      >
                        Layer-Wise
                      </button>
                    </div>
                  </div>
                </div>

                <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950/60 space-y-3">
                  <div className="text-neutral-300 font-bold text-[11px] uppercase tracking-wider text-cyan-400">
                    2. Recovery / Fine-Tuning
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-neutral-400">Enable Multimodal Recovery</span>
                    <input
                      type="checkbox"
                      checked={recoveryEnabled}
                      onChange={e => setRecoveryEnabled(e.target.checked)}
                      className="rounded accent-cyan-500"
                    />
                  </div>

                  {recoveryEnabled && (
                    <>
                      <div>
                        <label className="block text-neutral-400 mb-1">Method</label>
                        <select
                          value={trainingMethod}
                          onChange={e => setTrainingMethod(e.target.value as any)}
                          className="w-full px-2.5 py-1.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-200"
                        >
                          <option value="LoRA">LoRA (Low-Rank Adaptation)</option>
                          <option value="Adapter Only">Adapter / Projector Only</option>
                          <option value="Full Fine-Tuning">Full Fine-Tuning</option>
                        </select>
                      </div>

                      <div className="grid grid-cols-2 gap-2">
                        <div>
                          <label className="block text-neutral-400 mb-1">LoRA Rank</label>
                          <input
                            type="number"
                            value={loraRank}
                            onChange={e => setLoraRank(parseInt(e.target.value) || 16)}
                            className="w-full px-2.5 py-1.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-200"
                          />
                        </div>
                        <div>
                          <label className="block text-neutral-400 mb-1">Epochs</label>
                          <input
                            type="number"
                            value={epochs}
                            onChange={e => setEpochs(parseInt(e.target.value) || 1)}
                            className="w-full px-2.5 py-1.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-200"
                          />
                        </div>
                      </div>
                    </>
                  )}
                </div>
              </div>

              <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950/60 space-y-3">
                <div className="text-neutral-300 font-bold text-[11px] uppercase tracking-wider text-cyan-400">
                  3. Quantization &amp; AWQ Settings
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {(['None', 'LLM.int8', 'AWQ INT4'] as QuantizationMethod[]).map(m => (
                    <button
                      key={m}
                      type="button"
                      onClick={() => setQuantMethod(m)}
                      className={`py-2 rounded-lg border text-center font-bold ${
                        quantMethod === m
                          ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300'
                          : 'border-neutral-800 text-neutral-400 hover:text-neutral-300'
                      }`}
                    >
                      {m}
                    </button>
                  ))}
                </div>

                {quantMethod === 'AWQ INT4' && (
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-2">
                    <div>
                      <label className="block text-neutral-400 mb-1">Calibration Dataset</label>
                      <select
                        value={calibDataset}
                        onChange={e => setCalibDataset(e.target.value as any)}
                        className="w-full px-2.5 py-1.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-200"
                      >
                        <option value="ShareGPT4V">ShareGPT4V (Multimodal)</option>
                        <option value="COCO-QA">COCO-QA</option>
                        <option value="Pile-val">Pile-val (Text)</option>
                        <option value="Wikitext-2">Wikitext-2</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-neutral-400 mb-1">Samples</label>
                      <select
                        value={calibSamples}
                        onChange={e => setCalibSamples(parseInt(e.target.value))}
                        className="w-full px-2.5 py-1.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-200"
                      >
                        <option value={32}>32 samples (Fast)</option>
                        <option value={64}>64 samples</option>
                        <option value={128}>128 samples (Standard)</option>
                        <option value={256}>256 samples (Deep)</option>
                      </select>
                    </div>

                    <div>
                      <label className="block text-neutral-400 mb-1">Group Size</label>
                      <select
                        value={groupSize}
                        onChange={e => setGroupSize(parseInt(e.target.value) as any)}
                        className="w-full px-2.5 py-1.5 rounded bg-neutral-900 border border-neutral-700 text-neutral-200"
                      >
                        <option value={128}>128 (Standard AWQ)</option>
                        <option value={64}>64 (Higher Precision)</option>
                      </select>
                    </div>
                  </div>
                )}
              </div>

              <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950/60 flex items-center justify-between">
                <div>
                  <span className="text-neutral-300 font-bold block">Execution Worker</span>
                  <span className="text-neutral-500 text-[10px]">Select cloud GPU for recovery runs that exceed local 6GB VRAM</span>
                </div>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => setExecTarget('LOCAL')}
                    className={`px-3 py-1 rounded border ${
                      execTarget === 'LOCAL'
                        ? 'bg-neutral-800 border-neutral-600 text-neutral-200'
                        : 'border-neutral-800 text-neutral-500'
                    }`}
                  >
                    Local RTX 3050
                  </button>
                  <button
                    type="button"
                    onClick={() => setExecTarget('CLOUD')}
                    className={`px-3 py-1 rounded border ${
                      execTarget === 'CLOUD'
                        ? 'bg-cyan-500/20 border-cyan-500/50 text-cyan-300 font-bold'
                        : 'border-neutral-800 text-neutral-500'
                    }`}
                  >
                    Cloud GPU
                  </button>
                </div>
              </div>

              <div className="flex items-center justify-end gap-3 pt-3 border-t border-neutral-800">
                <button
                  type="button"
                  onClick={() => setShowConfigModal(false)}
                  className="px-4 py-2 rounded-xl border border-neutral-800 text-neutral-400 hover:text-neutral-200"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  className="px-5 py-2 rounded-xl bg-cyan-500 hover:bg-cyan-400 text-neutral-950 font-bold transition-colors"
                >
                  Register Experiment
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
};
