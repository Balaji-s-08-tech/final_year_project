import React, { useState } from 'react';
import { ResearchBadge } from '../common/Badge';
import {
  BarChart3,
  CheckCircle2,
  AlertCircle,
  Play,
  Layers,
  FileQuestion,
  Cpu,
  Clock,
  Zap,
  Activity
} from 'lucide-react';

interface BenchmarksViewProps {
  onRunBenchmarkDemo: () => void;
  isRunning: boolean;
}

export const BenchmarksView: React.FC<BenchmarksViewProps> = ({ onRunBenchmarkDemo, isRunning }) => {
  const [selectedSuite, setSelectedSuite] = useState<string>('pope');

  const benchmarkSuites = [
    {
      id: 'pope',
      name: 'POPE (Object Hallucination)',
      target: 'Multimodal Fidelity',
      questions: '3,000 Questions (Random, Popular, Adversarial)',
      metric: 'Accuracy, F1-Score, Precision',
      status: 'PENDING',
      desc: 'Evaluates whether compressed vision-language models fabricate non-existent objects when presented with deceptive or biased queries.'
    },
    {
      id: 'scienceqa',
      name: 'ScienceQA (Visual Reasoning)',
      target: 'Reasoning & Commonsense',
      questions: '2,017 Multimodal Questions',
      metric: 'Accuracy (IMG split)',
      status: 'PENDING',
      desc: 'Assesses multi-step visual reasoning and scientific diagram comprehension under compressed language backbone channels.'
    },
    {
      id: 'mme',
      name: 'MME Benchmark',
      target: 'Perception & Cognition',
      questions: '14 Sub-Tasks (Existence, Count, Position, Color)',
      metric: 'Perception & Cognition Cumulative Score',
      status: 'PENDING',
      desc: 'Holistic multimodal assessment testing basic perception tasks and complex cognitive reasoning.'
    },
    {
      id: 'textvqa',
      name: 'TextVQA (Visual Text OCR)',
      target: 'OCR & Cross-Modal Grounding',
      questions: '5,000 Evaluation Pairs',
      metric: 'VQA Accuracy (%)',
      status: 'PENDING',
      desc: 'Measures reading capabilities from scene text images; sensitive to vision encoder and cross-modal projector alignment.'
    },
    {
      id: 'hardware',
      name: 'Hardware Latency & Throughput',
      target: 'Edge Profiling',
      questions: 'Batch sizes 1, 4, 8 (Sequence len 512, 1024)',
      metric: 'Time-to-First-Token (ms), Tokens/sec, Peak VRAM',
      status: 'PENDING',
      desc: 'Measures physical memory throughput on target NVIDIA GPUs (RTX 3050 and Cloud A100/T4).'
    }
  ];

  return (
    <div id="benchmarks-view" className="space-y-6">
      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider">
              Evaluation Protocols
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-amber-950/60 border border-amber-800/60 text-amber-300 font-mono">
              Pending Execution
            </span>
          </div>
          <h2 className="text-lg font-bold text-neutral-100">Downstream Multimodal Benchmark Harness</h2>
          <p className="text-xs text-neutral-400 mt-0.5 max-w-2xl">
            Standardized evaluation suites measuring object hallucination, reasoning fidelity, and hardware execution efficiency.
          </p>
        </div>

        <button
          onClick={onRunBenchmarkDemo}
          disabled={isRunning}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-mono text-xs font-semibold transition-all ${
            isRunning
              ? 'bg-neutral-800 text-neutral-500 cursor-not-allowed'
              : 'bg-cyan-500 hover:bg-cyan-400 text-neutral-950 font-bold shadow-[0_0_15px_rgba(6,182,212,0.25)]'
          }`}
        >
          <Play className="w-3.5 h-3.5 fill-current" />
          <span>{isRunning ? 'Running Profiler...' : 'Launch Benchmark Harness (Demo)'}</span>
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="space-y-3">
          <div className="text-xs font-mono text-neutral-400 uppercase tracking-wider px-1">
            Evaluation Suites
          </div>
          {benchmarkSuites.map(suite => {
            const isSelected = selectedSuite === suite.id;
            return (
              <button
                key={suite.id}
                onClick={() => setSelectedSuite(suite.id)}
                className={`w-full text-left p-4 rounded-xl border transition-all ${
                  isSelected
                    ? 'bg-neutral-800 border-cyan-500/60 shadow-[0_0_15px_rgba(6,182,212,0.1)]'
                    : 'bg-neutral-900/60 border-neutral-800 hover:border-neutral-700'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-bold text-neutral-100">{suite.name}</span>
                  <ResearchBadge status={suite.status as any} size="sm" />
                </div>
                <div className="text-[11px] font-mono text-cyan-400">{suite.target}</div>
                <div className="text-[11px] text-neutral-400 mt-1 line-clamp-2 leading-relaxed">
                  {suite.desc}
                </div>
              </button>
            );
          })}
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
                <BarChart3 className="w-4 h-4 text-cyan-400" />
                <span>Suite Evaluation Configuration &amp; Protocol</span>
              </h3>
              <ResearchBadge status="PENDING" />
            </div>

            {selectedSuite === 'pope' && (
              <div className="space-y-4 text-xs font-mono">
                <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-950 space-y-2">
                  <div className="text-neutral-200 font-bold text-sm">POPE Protocol Details</div>
                  <p className="font-sans text-xs text-neutral-400 leading-relaxed">
                    Evaluates visual hallucination across MSCOCO image validation sets. Binary queries in the form: <em>&quot;Is there a &lt;object&gt; in the image?&quot;</em> are categorized into Random, Popular, and Adversarial distributions to test whether compressed models suffer from degraded language prior grounding.
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/60">
                    <span className="text-neutral-500 text-[11px] block">Questions Count</span>
                    <span className="text-neutral-200 font-bold text-sm">3,000 Questions</span>
                  </div>
                  <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/60">
                    <span className="text-neutral-500 text-[11px] block">Target Metrics</span>
                    <span className="text-neutral-200 font-bold text-sm">Accuracy, F1, Yes-Ratio</span>
                  </div>
                </div>
              </div>
            )}

            {selectedSuite === 'scienceqa' && (
              <div className="space-y-4 text-xs font-mono">
                <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-950 space-y-2">
                  <div className="text-neutral-200 font-bold text-sm">ScienceQA Protocol Details</div>
                  <p className="font-sans text-xs text-neutral-400 leading-relaxed">
                    Evaluates multi-modal science question answering across elementary and high-school topics. Questions contain multimodal contexts (diagrams, photos) combined with text questions and multiple-choice options.
                  </p>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/60">
                    <span className="text-neutral-500 text-[11px] block">Test Set Split</span>
                    <span className="text-neutral-200 font-bold text-sm">2,017 Image-Context Problems</span>
                  </div>
                  <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/60">
                    <span className="text-neutral-500 text-[11px] block">Chain of Thought</span>
                    <span className="text-neutral-200 font-bold text-sm">Direct Answer Accuracy</span>
                  </div>
                </div>
              </div>
            )}

            {selectedSuite === 'hardware' && (
              <div className="space-y-4 text-xs font-mono">
                <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-950 space-y-2">
                  <div className="text-neutral-200 font-bold text-sm">Hardware Profiling Harness</div>
                  <p className="font-sans text-xs text-neutral-400 leading-relaxed">
                    Measures CUDA runtime execution characteristics: Time to First Token (TTFT), sustained token generation throughput (tokens/second), and peak VRAM allocation across batch sizes 1 and 4 using PyTorch CUDA events and nvidia-smi telemetry.
                  </p>
                </div>
                <div className="grid grid-cols-3 gap-3">
                  <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/60">
                    <span className="text-neutral-500 text-[11px] block">Local Device</span>
                    <span className="text-neutral-200 font-bold text-xs">RTX 3050 (6GB)</span>
                  </div>
                  <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/60">
                    <span className="text-neutral-500 text-[11px] block">Cloud Device</span>
                    <span className="text-neutral-200 font-bold text-xs">NVIDIA A100 (40GB)</span>
                  </div>
                  <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/60">
                    <span className="text-neutral-500 text-[11px] block">Kernel</span>
                    <span className="text-cyan-300 font-bold text-xs">W4A16 GEMM</span>
                  </div>
                </div>
              </div>
            )}

            <div className="p-4 rounded-xl border border-neutral-800/80 bg-neutral-950/50 text-xs text-neutral-400 space-y-1 font-sans">
              <strong className="text-neutral-300 font-mono">Current Status:</strong>
              <p className="text-[11px] leading-relaxed">
                Empirical evaluation runs will be executed once the cloud recovery pipeline yields the complete recovered checkpoint. In compliance with research integrity standards, no benchmark scores are fabricated.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
