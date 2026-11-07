import React from 'react';
import { ResearchBadge } from '../common/Badge';
import {
  Binary,
  Sparkles,
  ArrowDown,
  ShieldCheck,
  Zap,
  HelpCircle,
  Cpu,
  Layers,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';

export const QuantizationView: React.FC = () => {
  const awqSteps = [
    {
      step: '01',
      title: 'Multimodal Activation Statistics',
      desc: 'Weights W are paired with activation statistics X collected by running calibration forward passes through the vision-language network.',
      badge: 'Input Matrix'
    },
    {
      step: '02',
      title: 'Salient Channel Detection',
      desc: 'Identifies top 1% channels with disproportionately large activation magnitudes that dictate output token fidelity.',
      badge: 'Saliency Profiling'
    },
    {
      step: '03',
      title: 'Activation-Aware Per-Channel Scaling',
      desc: 'Multiplies weights by protective scaling factor s while dividing activations by s (W′ = W · diag(s)⁻¹, X′ = diag(s) · X) to preserve the exact mathematical product.',
      badge: 'Error Minimization'
    },
    {
      step: '04',
      title: 'INT4 Weight Quantization & Packing',
      desc: 'Weights are rounded to 4-bit integers and packed into 32-bit registers (group size 128), slashing memory bandwidth by 3.8x.',
      badge: 'W4A16 Packed'
    }
  ];

  return (
    <div id="quantization-view" className="space-y-6">
      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider">
              Quantization Methodology
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/60 text-cyan-300 font-mono">
              AWQ INT4 (W4A16)
            </span>
          </div>
          <h2 className="text-lg font-bold text-neutral-100">
            Activation-Aware Weight Quantization (AWQ) Theory &amp; Integration
          </h2>
          <p className="text-xs text-neutral-400 mt-0.5 max-w-2xl">
            Protecting salient weights identified through multimodal activation dynamics before aggressive 4-bit integer quantization.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <ResearchBadge status="VERIFIED" />
        </div>
      </div>

      <div className="p-6 rounded-2xl border border-neutral-800 bg-neutral-950/70 space-y-6">
        <div className="text-xs font-mono text-neutral-500 uppercase tracking-widest text-center">
          ─── AWQ QUANTIZATION MECHANISM ───
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          {awqSteps.map((s, idx) => (
            <div
              key={s.step}
              className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/60 flex flex-col justify-between space-y-3 relative"
            >
              <div>
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-mono font-bold text-cyan-400">Step {s.step}</span>
                  <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
                    {s.badge}
                  </span>
                </div>
                <h4 className="text-xs font-bold text-neutral-200">{s.title}</h4>
                <p className="text-xs text-neutral-400 mt-1.5 leading-relaxed">{s.desc}</p>
              </div>

              {idx < awqSteps.length - 1 && (
                <div className="hidden lg:block absolute -right-3 top-1/2 -translate-y-1/2 z-10 text-neutral-600 font-mono">
                  →
                </div>
              )}
            </div>
          ))}
        </div>

        <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/40 text-xs font-mono text-neutral-300 space-y-2">
          <div className="text-cyan-300 font-bold text-[11px] uppercase tracking-wider">
            Key Insight: Why AWQ After Structured Pruning?
          </div>
          <p className="text-xs font-sans text-neutral-400 leading-relaxed">
            Structured pruning physically eliminates low-sensitivity channels to permanently cut FLOPS and memory size. However, the surviving dense matrices still store weights in 16-bit floating point. AWQ operates orthogonally on these reduced matrices: by preserving salient activation channels, it compresses the remaining parameters into packed 4-bit integers without the severe perceptual degradation of naive Round-To-Nearest (RTN) quantization.
          </p>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-neutral-800">
            <HelpCircle className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-neutral-100">Research Question 1</h3>
          </div>
          <div className="text-xs font-semibold text-neutral-200 leading-snug">
            “Does structural pruning change the activation-aware importance landscape used by AWQ?”
          </div>
          <p className="text-xs text-neutral-400 leading-relaxed font-sans">
            In unpruned models, ~1% of channels exhibit extreme activation outlier magnitudes. When 20% of channels are eliminated via first-order Taylor criteria, remaining channels must absorb compensatory representation. We investigate whether this redistribution produces new activation outliers that require recalibrated AWQ protection scales rather than applying static baseline parameters.
          </p>
          <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950 font-mono text-[11px] text-neutral-400 flex items-center justify-between">
            <span>Status: Calibration profiling active</span>
            <ResearchBadge status="VERIFIED" size="sm" />
          </div>
        </div>

        <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
          <div className="flex items-center gap-2 pb-3 border-b border-neutral-800">
            <HelpCircle className="w-4 h-4 text-cyan-400" />
            <h3 className="text-sm font-bold text-neutral-100">Research Question 2</h3>
          </div>
          <div className="text-xs font-semibold text-neutral-200 leading-snug">
            “At what pruning ratio does AWQ retain an advantage over the INT8 baseline?”
          </div>
          <p className="text-xs text-neutral-400 leading-relaxed font-sans">
            LLM.int8 maintains 8-bit precision with mixed-precision outlier preservation. AWQ pushes compression to 4 bits with per-channel scaling. We evaluate if aggressive structural pruning (e.g., &gt;30%) compounds with 4-bit quantization to cause catastrophic failure, or whether AWQ preserves the Pareto frontier across 10%, 20%, and 30% reduction ratios.
          </p>
          <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950 font-mono text-[11px] text-neutral-400 flex items-center justify-between">
            <span>Status: Downstream benchmark pending</span>
            <ResearchBadge status="PENDING" size="sm" />
          </div>
        </div>
      </div>

      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
          <div>
            <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
              <Binary className="w-4 h-4 text-cyan-400" />
              <span>Quantization Regimes Comparison</span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Structural and operational distinctions between standard baseline precision regimes.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-[11px]">
                <th className="py-2.5 px-3">Regime</th>
                <th className="py-2.5 px-3">Bits (W/A)</th>
                <th className="py-2.5 px-3">Outlier Handling</th>
                <th className="py-2.5 px-3">Hardware Memory Footprint</th>
                <th className="py-2.5 px-3 text-right">Pipeline Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-850">
              <tr className="hover:bg-neutral-800/30">
                <td className="py-3 px-3 font-semibold text-neutral-200">FP16 Baseline</td>
                <td className="py-3 px-3 text-neutral-400">16-bit / 16-bit</td>
                <td className="py-3 px-3 text-neutral-400">Full floating point dynamic range</td>
                <td className="py-3 px-3 text-neutral-300">~6.24 GB (Base Checkpoint)</td>
                <td className="py-3 px-3 text-right"><ResearchBadge status="VERIFIED" size="sm" /></td>
              </tr>
              <tr className="hover:bg-neutral-800/30">
                <td className="py-3 px-3 font-semibold text-neutral-200">LLM.int8</td>
                <td className="py-3 px-3 text-neutral-400">8-bit / 8-bit</td>
                <td className="py-3 px-3 text-neutral-400">Mixed precision FP16 outliers (&gt;6.0)</td>
                <td className="py-3 px-3 text-neutral-300">~3.20 GB</td>
                <td className="py-3 px-3 text-right"><ResearchBadge status="PENDING" size="sm" /></td>
              </tr>
              <tr className="hover:bg-neutral-800/30 bg-cyan-950/20">
                <td className="py-3 px-3 font-bold text-cyan-300">AWQ INT4 (Proposed)</td>
                <td className="py-3 px-3 text-cyan-400 font-semibold">4-bit / 16-bit</td>
                <td className="py-3 px-3 text-neutral-200">Per-channel activation-aware scaling</td>
                <td className="py-3 px-3 text-cyan-300 font-bold">~1.65 GB (Projected)</td>
                <td className="py-3 px-3 text-right"><ResearchBadge status="VERIFIED" size="sm" /></td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
