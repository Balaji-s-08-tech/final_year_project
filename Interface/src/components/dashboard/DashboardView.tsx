import React, { useState } from 'react';
import { PipelineStageInfo, VerifiedDevelopmentItem } from '../../types/research';
import { StatCard } from '../common/StatCard';
import { ResearchBadge, ExecutionBadge } from '../common/Badge';
import {
  GitCommit,
  CheckCircle2,
  AlertTriangle,
  Layers,
  ArrowRight,
  Info,
  Clock,
  Sparkles,
  Zap,
  Play
} from 'lucide-react';

interface DashboardViewProps {
  stages: PipelineStageInfo[];
  verifiedItems: VerifiedDevelopmentItem[];
  onOpenStageDetail: (stage: PipelineStageInfo) => void;
  onNavigate: (view: any) => void;
  onRunQuickDemo: () => void;
  isRunning: boolean;
}

export const DashboardView: React.FC<DashboardViewProps> = ({
  stages,
  verifiedItems,
  onOpenStageDetail,
  onNavigate,
  onRunQuickDemo,
  isRunning
}) => {
  const [selectedStage, setSelectedStage] = useState<PipelineStageInfo>(stages[0]);

  const verifiedCount = verifiedItems.filter(i => i.status === 'VERIFIED').length;
  const blockedCount = verifiedItems.filter(i => i.status === 'BLOCKED').length;
  const pendingCount = verifiedItems.filter(i => i.status === 'PENDING').length;

  const corePipelineSummary = [
    { key: 'model_loading', label: 'Original Bunny', stage: stages[0] },
    { key: 'structural_pruning', label: 'Structured Pruning', stage: stages[3] },
    { key: 'recovery', label: 'Recovery (LoRA)', stage: stages[4] },
    { key: 'weight_quantization', label: 'AWQ INT4', stage: stages[6] },
    { key: 'evaluation', label: 'Evaluation', stage: stages[9] }
  ];

  return (
    <div id="dashboard-view" className="space-y-6">
      <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 p-5 rounded-2xl border border-neutral-800 bg-gradient-to-r from-neutral-900/90 via-neutral-900/60 to-neutral-950">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider">
              Research Objective
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 font-mono">
              Final Year Dissertation
            </span>
          </div>
          <h2 className="text-lg sm:text-xl font-bold text-neutral-100 tracking-tight">
            Structured Pruning &amp; Activation-Aware Weight Quantization (AWQ)
          </h2>
          <p className="text-xs sm:text-sm text-neutral-400 mt-1 max-w-3xl leading-relaxed">
            Investigating compound compression dynamics on the Bunny-v1_0-3B multimodal architecture (SigLIP-SO400M vision encoder + Phi-2 language backbone) under tight hardware constraints.
          </p>
        </div>

        <div className="flex items-center gap-2.5 shrink-0">
          <button
            id="btn-quick-run-pipeline"
            onClick={onRunQuickDemo}
            disabled={isRunning}
            className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-mono text-xs font-semibold transition-all ${
              isRunning
                ? 'bg-cyan-950 text-cyan-400 border border-cyan-800 cursor-not-allowed animate-pulse'
                : 'bg-cyan-500 hover:bg-cyan-400 text-neutral-950 font-bold shadow-[0_0_20px_rgba(6,182,212,0.3)]'
            }`}
          >
            <Play className="w-3.5 h-3.5 fill-current" />
            <span>{isRunning ? 'Pipeline Running...' : 'Execute Pipeline (Demo)'}</span>
          </button>
          <button
            onClick={() => onNavigate('presentation')}
            className="px-3.5 py-2.5 rounded-xl border border-neutral-700 bg-neutral-800/80 hover:bg-neutral-800 text-neutral-200 text-xs font-mono transition-colors"
          >
            Review Deck
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Base Parameters"
          value="3.12B"
          unit="weights"
          status="VERIFIED"
          note="SigLIP (428M) + Phi-2 (2.78B)"
          icon={<Layers className="w-4 h-4 text-cyan-400" />}
        />
        <StatCard
          title="Pruned Target"
          value="~2.56B"
          unit="parameters"
          status="VERIFIED"
          note="Taylor FO 20% width reduction verified"
          icon={<GitCommit className="w-4 h-4 text-emerald-400" />}
          highlight={true}
        />
        <StatCard
          title="Quantization Format"
          value="INT4 (W4A16)"
          unit="AWQ Packed"
          status="VERIFIED"
          note="Layer replacement verified in smoke test"
          icon={<Zap className="w-4 h-4 text-cyan-400" />}
        />
        <StatCard
          title="Multimodal Recovery"
          value="Hardware Blocked"
          unit="Local RTX 3050"
          status="BLOCKED"
          note="Requires >14GB VRAM; Cloud scheduled"
          icon={<AlertTriangle className="w-4 h-4 text-rose-400" />}
        />
      </div>

      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between pb-4 border-b border-neutral-800 gap-2">
          <div>
            <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
              <span>Main Research Pipeline</span>
              <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
                Interactive Stage Visualizer
              </span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Click any stage node to inspect architectural transformations, status, and telemetry.
            </p>
          </div>
          <button
            onClick={() => onNavigate('pipeline')}
            className="text-xs font-mono text-cyan-400 hover:text-cyan-300 flex items-center gap-1 self-start sm:self-auto"
          >
            <span>Full 10-Stage Architecture</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-5 gap-3 pt-4">
          {corePipelineSummary.map((item, idx) => {
            const isSelected = selectedStage.key === item.stage.key;
            return (
              <div key={item.key} className="relative flex flex-col">
                <button
                  onClick={() => {
                    setSelectedStage(item.stage);
                    onOpenStageDetail(item.stage);
                  }}
                  className={`w-full text-left p-3.5 rounded-xl border transition-all flex flex-col justify-between h-full ${
                    isSelected
                      ? 'bg-neutral-800 border-cyan-500/60 shadow-[0_0_15px_rgba(6,182,212,0.15)]'
                      : 'bg-neutral-950/70 border-neutral-800 hover:border-neutral-700'
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-mono text-neutral-500 font-semibold">
                        Stage 0{item.stage.id}
                      </span>
                      <ExecutionBadge status={item.stage.status} />
                    </div>
                    <div className="text-xs font-bold text-neutral-200 line-clamp-1">{item.label}</div>
                    <div className="text-[11px] text-neutral-400 mt-1 line-clamp-2 leading-snug">
                      {item.stage.name}
                    </div>
                  </div>

                  <div className="mt-3 pt-2.5 border-t border-neutral-800/80 flex items-center justify-between text-[10px] font-mono">
                    <span className="text-neutral-500 capitalize">{item.stage.executionType}</span>
                    <ResearchBadge status={item.stage.researchStatus} size="sm" />
                  </div>
                </button>

                {idx < corePipelineSummary.length - 1 && (
                  <div className="hidden md:flex absolute -right-2 top-1/2 -translate-y-1/2 z-10 text-neutral-600 pointer-events-none">
                    →
                  </div>
                )}
              </div>
            );
          })}
        </div>

        <div className="mt-4 p-3.5 rounded-xl border border-neutral-800/80 bg-neutral-950/60 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
          <div className="flex items-center gap-2 text-neutral-300">
            <Info className="w-4 h-4 text-cyan-400 shrink-0" />
            <span>
              <strong className="text-neutral-100">Active Selection: </strong>
              {selectedStage.name} — {selectedStage.detail}
            </span>
          </div>
          <div className="flex items-center gap-2 font-mono text-[11px] shrink-0">
            <span className="text-neutral-500">Progress: {selectedStage.progress}%</span>
            <div className="w-20 h-2 bg-neutral-800 rounded-full overflow-hidden">
              <div
                className="h-full bg-cyan-400 transition-all duration-300"
                style={{ width: `${selectedStage.progress}%` }}
              />
            </div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
            <div>
              <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                <span>Verified Implementation Milestones</span>
              </h3>
              <p className="text-xs text-neutral-400 mt-0.5">
                Current progress validated against research repository code.
              </p>
            </div>
            <div className="flex items-center gap-2 text-[10px] font-mono">
              <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
                {verifiedCount} Verified
              </span>
              <span className="px-2 py-0.5 rounded bg-rose-500/10 text-rose-400 border border-rose-500/30">
                {blockedCount} Blocked
              </span>
              <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-300 border border-amber-500/30">
                {pendingCount} Pending
              </span>
            </div>
          </div>

          <div className="space-y-2.5">
            {verifiedItems.slice(0, 5).map(item => (
              <div
                key={item.id}
                className="p-3 rounded-xl border border-neutral-800/80 bg-neutral-950/50 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 hover:border-neutral-700 transition-colors"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-mono text-neutral-500">{item.id}</span>
                    <h4 className="text-xs font-semibold text-neutral-200">{item.title}</h4>
                    <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-neutral-900 border border-neutral-800 text-neutral-400">
                      {item.component}
                    </span>
                  </div>
                  <p className="text-[11px] text-neutral-400 leading-snug">{item.detail}</p>
                </div>
                <div className="shrink-0 flex sm:flex-col items-end justify-between gap-1">
                  <ResearchBadge status={item.status} size="sm" />
                  <span className="text-[10px] font-mono text-neutral-500">{item.timestamp}</span>
                </div>
              </div>
            ))}
          </div>

          <div className="pt-2 text-center">
            <button
              onClick={() => onNavigate('system')}
              className="text-xs font-mono text-cyan-400 hover:text-cyan-300 inline-flex items-center gap-1"
            >
              <span>View Full Hardware &amp; Verification Matrix</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>

        <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center gap-2 pb-3 border-b border-neutral-800">
              <Sparkles className="w-4 h-4 text-cyan-400" />
              <h3 className="text-sm font-bold text-neutral-100">Core Research Inquiries</h3>
            </div>

            <div className="mt-4 space-y-3.5">
              <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/70">
                <div className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider font-semibold mb-1">
                  Research Question 1
                </div>
                <div className="text-xs font-medium text-neutral-200 leading-snug">
                  “Does structural pruning change the activation-aware importance landscape used by AWQ?”
                </div>
                <div className="text-[11px] text-neutral-400 mt-1.5 leading-relaxed">
                  Investigating whether channel removal alters activation outlier concentrations, requiring recalibrated scales.
                </div>
              </div>

              <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/70">
                <div className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider font-semibold mb-1">
                  Research Question 2
                </div>
                <div className="text-xs font-medium text-neutral-200 leading-snug">
                  “At what pruning ratio does AWQ retain an advantage over the INT8 baseline?”
                </div>
                <div className="text-[11px] text-neutral-400 mt-1.5 leading-relaxed">
                  Determining the compound Pareto efficiency curve across 10%, 20%, and 30% width-pruned checkpoints.
                </div>
              </div>
            </div>
          </div>

          <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950/80 text-xs space-y-2">
            <div className="flex items-center justify-between text-neutral-300 font-medium">
              <span>Local Hardware Footprint</span>
              <span className="font-mono text-cyan-400">RTX 3050 (6GB)</span>
            </div>
            <div className="space-y-1">
              <div className="flex justify-between text-[11px] font-mono text-neutral-400">
                <span>VRAM Allocation</span>
                <span>2.4 / 6.0 GB (40%)</span>
              </div>
              <div className="w-full h-1.5 bg-neutral-800 rounded-full overflow-hidden">
                <div className="h-full bg-cyan-400 w-[40%]" />
              </div>
            </div>
            <div className="text-[10px] font-mono text-neutral-500 pt-1">
              Recommended: <span className="text-neutral-300 font-semibold">HYBRID MODE</span> for full recovery training
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
