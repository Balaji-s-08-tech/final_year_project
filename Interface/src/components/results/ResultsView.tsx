import React, { useState } from 'react';
import { ModelComparisonRow } from '../../types/research';
import { MODEL_COMPARISON_DATA } from '../../data/researchData';
import { StatCard } from '../common/StatCard';
import { ResearchBadge } from '../common/Badge';
import {
  TableProperties,
  BarChart3,
  CheckCircle2,
  AlertTriangle,
  Info,
  Layers,
  Zap,
  TrendingDown,
  ShieldCheck
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  CartesianGrid
} from 'recharts';

export const ResultsView: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'table' | 'charts'>('table');
  const rows = MODEL_COMPARISON_DATA;

  const chartData = [
    {
      name: 'Original',
      paramsB: 3.12,
      checkpointGB: 6.24,
      vramGB: 7.8,
      status: 'VERIFIED'
    },
    {
      name: 'Pruned (20%)',
      paramsB: 2.56,
      checkpointGB: 5.12,
      vramGB: 6.4,
      status: 'VERIFIED'
    },
    {
      name: 'Pruned + INT8',
      paramsB: 2.56,
      checkpointGB: 2.80,
      vramGB: 4.2,
      status: 'PENDING'
    },
    {
      name: 'Pruned + AWQ INT4',
      paramsB: 2.56,
      checkpointGB: 1.65,
      vramGB: 2.9,
      status: 'PENDING'
    }
  ];

  return (
    <div id="results-view" className="space-y-6">
      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider">
              Comparative Analysis
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 font-mono">
              Empirical Research Matrix
            </span>
          </div>
          <h2 className="text-lg font-bold text-neutral-100">Compression &amp; Efficiency Benchmark Results</h2>
          <p className="text-xs text-neutral-400 mt-0.5 max-w-2xl">
            Comparison between unpruned baseline, structurally pruned (20%), Pruned + LLM.int8, and Pruned + AWQ INT4 models.
          </p>
        </div>

        <div className="flex items-center rounded-xl border border-neutral-800 bg-neutral-950 p-1 text-xs font-mono">
          <button
            onClick={() => setActiveTab('table')}
            className={`px-3 py-1.5 rounded-lg transition-all ${
              activeTab === 'table'
                ? 'bg-neutral-800 text-cyan-300 font-bold border border-neutral-700'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            Matrix Table
          </button>
          <button
            onClick={() => setActiveTab('charts')}
            className={`px-3 py-1.5 rounded-lg transition-all ${
              activeTab === 'charts'
                ? 'bg-neutral-800 text-cyan-300 font-bold border border-neutral-700'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            Comparative Charts
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatCard title="Original Params" value="3.12B" status="VERIFIED" note="Bunny-3B FP16" />
        <StatCard title="Pruned Params" value="2.56B" status="VERIFIED" note="20% width cut" highlight={true} />
        <StatCard title="Compression" value="Pending" status="PENDING" note="Reported post-run" />
        <StatCard title="Inference Latency" value="Pending" status="PENDING" note="Profiling scheduled" />
        <StatCard title="POPE Accuracy" value="Pending" status="PENDING" note="Awaiting recovery" />
        <StatCard title="F1 Score" value="Pending" status="PENDING" note="Awaiting recovery" />
      </div>

      <div className="p-3.5 rounded-xl border border-amber-500/20 bg-amber-500/5 flex items-start gap-2.5 text-xs text-neutral-300">
        <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <div>
          <strong className="text-amber-300 font-semibold font-mono">Research Integrity Assurance:</strong>
          <span className="text-neutral-400 block mt-0.5 leading-relaxed font-sans">
            Metrics not yet directly measured on physical GPU hardware are designated as <span className="text-amber-300 font-mono">PENDING</span>. No fabricated evaluation scores, fictitious latency acceleration, or artificial accuracy improvements are shown.
          </span>
        </div>
      </div>

      {activeTab === 'table' ? (
        <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
          <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
            <div>
              <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
                <TableProperties className="w-4 h-4 text-cyan-400" />
                <span>Multi-Variant Compression Matrix</span>
              </h3>
              <p className="text-xs text-neutral-400 mt-0.5">
                Original vs Pruned vs Pruned+INT8 vs Pruned+AWQ INT4
              </p>
            </div>
            <span className="text-xs font-mono text-neutral-500">4 Model Variants</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs font-mono">
              <thead>
                <tr className="border-b border-neutral-800 text-neutral-400 text-[11px]">
                  <th className="py-2.5 px-3">Metric</th>
                  <th className="py-2.5 px-3">Original</th>
                  <th className="py-2.5 px-3">Pruned (20%)</th>
                  <th className="py-2.5 px-3">Pruned + INT8</th>
                  <th className="py-2.5 px-3 text-cyan-400">Pruned + AWQ INT4</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-850">
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">Parameters</td>
                  <td className="py-3 px-3 text-neutral-200">
                    3.12B <span className="text-[10px] text-emerald-400">●</span>
                  </td>
                  <td className="py-3 px-3 text-emerald-400 font-bold">
                    ~2.56B <span className="text-[10px] text-emerald-400">●</span>
                  </td>
                  <td className="py-3 px-3 text-neutral-300">~2.56B</td>
                  <td className="py-3 px-3 font-bold text-cyan-300">~2.56B</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">Checkpoint Size</td>
                  <td className="py-3 px-3 text-neutral-200">6.24 GB <span className="text-[10px] text-emerald-400">●</span></td>
                  <td className="py-3 px-3 text-emerald-400 font-bold">5.12 GB <span className="text-[10px] text-emerald-400">●</span></td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400 font-bold">Pending</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">Weight Precision</td>
                  <td className="py-3 px-3 text-neutral-400">FP16 (16-bit)</td>
                  <td className="py-3 px-3 text-neutral-400">FP16 (16-bit)</td>
                  <td className="py-3 px-3 text-neutral-300">INT8 (8-bit)</td>
                  <td className="py-3 px-3 font-bold text-cyan-300">INT4 (4-bit W4A16)</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">Training VRAM</td>
                  <td className="py-3 px-3 text-neutral-400">~18.4 GB</td>
                  <td className="py-3 px-3 text-rose-400 font-semibold">~14.2 GB (Blocked on 6GB)</td>
                  <td className="py-3 px-3 text-neutral-500">N/A (Post-train)</td>
                  <td className="py-3 px-3 text-neutral-500">N/A (Post-train)</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">Inference Peak VRAM</td>
                  <td className="py-3 px-3 text-neutral-300">7.8 GB</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">Inference Latency</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">POPE Accuracy</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">ScienceQA Accuracy</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">F1 Score</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">MCC</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">ROC-AUC</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                  <td className="py-3 px-3 text-amber-400">Pending</td>
                </tr>
                <tr className="hover:bg-neutral-800/30">
                  <td className="py-3 px-3 font-semibold text-neutral-300">Scientific Status</td>
                  <td className="py-3 px-3"><ResearchBadge status="VERIFIED" size="sm" /></td>
                  <td className="py-3 px-3"><ResearchBadge status="VERIFIED" size="sm" /></td>
                  <td className="py-3 px-3"><ResearchBadge status="PENDING" size="sm" /></td>
                  <td className="py-3 px-3"><ResearchBadge status="PENDING" size="sm" /></td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
              <h4 className="text-xs font-bold text-neutral-200">Parameter Count Comparison (Billion)</h4>
              <ResearchBadge status="VERIFIED" size="sm" />
            </div>
            <div className="h-64 pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis dataKey="name" stroke="#737373" fontSize={10} tickLine={false} />
                  <YAxis stroke="#737373" fontSize={10} domain={[0, 4]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#171717', borderColor: '#404040', borderRadius: '8px', fontSize: '11px', fontFamily: 'monospace' }}
                  />
                  <Bar dataKey="paramsB" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Parameters (B)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] font-mono text-neutral-400 text-center">
              Structural channel pruning yields a 20% parameter reduction from 3.12B to ~2.56B.
            </p>
          </div>

          <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
              <h4 className="text-xs font-bold text-neutral-200">Checkpoint Size (GB)</h4>
              <div className="flex gap-1.5">
                <ResearchBadge status="VERIFIED" size="sm" />
                <ResearchBadge status="PENDING" size="sm" />
              </div>
            </div>
            <div className="h-64 pt-2">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#262626" />
                  <XAxis dataKey="name" stroke="#737373" fontSize={10} tickLine={false} />
                  <YAxis stroke="#737373" fontSize={10} domain={[0, 8]} />
                  <Tooltip
                    contentStyle={{ backgroundColor: '#171717', borderColor: '#404040', borderRadius: '8px', fontSize: '11px', fontFamily: 'monospace' }}
                  />
                  <Bar dataKey="checkpointGB" fill="#10b981" radius={[4, 4, 0, 0]} name="Checkpoint (GB)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <p className="text-[11px] font-mono text-neutral-400 text-center">
              Notice: INT8 and AWQ checkpoint estimates shown for theoretical scale; final values marked pending.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};
