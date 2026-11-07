import React, { useState } from 'react';
import { PROJECT_PRESENTATION_DATA, HARDWARE_SPEC } from '../../data/researchData';
import { ResearchBadge } from '../common/Badge';
import {
  Presentation,
  CheckCircle2,
  AlertTriangle,
  ArrowRight,
  ArrowLeft,
  X,
  Layers,
  Sparkles,
  Cpu,
  GitCommit,
  HelpCircle,
  ShieldAlert,
  ChevronRight
} from 'lucide-react';

interface PresentationModeProps {
  onClose: () => void;
  onNavigateView: (view: any) => void;
}

export const PresentationMode: React.FC<PresentationModeProps> = ({ onClose, onNavigateView }) => {
  const [slideIndex, setSlideIndex] = useState<number>(0);
  const data = PROJECT_PRESENTATION_DATA;

  const slides = [
    {
      id: 'overview',
      title: 'Project Overview & Research Focus',
      tag: 'Slide 1 of 6',
      content: (
        <div className="space-y-6">
          <div className="p-6 rounded-2xl border border-cyan-500/30 bg-neutral-900/90 shadow-[0_0_30px_rgba(6,182,212,0.15)] space-y-3">
            <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs uppercase tracking-widest font-semibold">
              Final Year Research Dissertation
            </div>
            <h1 className="text-xl sm:text-2xl font-extrabold text-neutral-100 leading-tight">
              {data.title}
            </h1>
            <p className="text-sm text-neutral-300 leading-relaxed font-sans pt-1">
              Investigating the compound dynamics of physical channel pruning and 4-bit activation-aware quantization on compact multimodal foundation models.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/60 space-y-1.5 font-mono text-xs">
              <span className="text-neutral-500 text-[11px] uppercase tracking-wider block">Target Architecture</span>
              <span className="text-neutral-100 font-bold text-sm block">Bunny-v1_0-3B</span>
              <span className="text-neutral-400 text-[11px] block font-sans">SigLIP (428M) + Phi-2 (2.78B)</span>
            </div>
            <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/60 space-y-1.5 font-mono text-xs">
              <span className="text-neutral-500 text-[11px] uppercase tracking-wider block">Core Pipeline</span>
              <span className="text-cyan-300 font-bold text-sm block">Taylor Prune → AWQ INT4</span>
              <span className="text-neutral-400 text-[11px] block font-sans">Width reduction + W4A16 GEMM</span>
            </div>
            <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/60 space-y-1.5 font-mono text-xs">
              <span className="text-neutral-500 text-[11px] uppercase tracking-wider block">Execution Strategy</span>
              <span className="text-amber-300 font-bold text-sm block">HYBRID (Local + Cloud)</span>
              <span className="text-neutral-400 text-[11px] block font-sans">RTX 3050 (6GB) + Cloud Worker</span>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'problem',
      title: 'Problem Statement & Existing Approach',
      tag: 'Slide 2 of 6',
      content: (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-6 rounded-2xl border border-neutral-800 bg-neutral-900/70 space-y-4">
            <div className="flex items-center gap-2 text-rose-400 font-mono text-xs font-bold uppercase tracking-wider">
              <AlertTriangle className="w-4 h-4" />
              <span>The Research Problem</span>
            </div>
            <p className="text-xs sm:text-sm text-neutral-300 leading-relaxed font-sans">
              {data.researchProblem}
            </p>
            <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950 text-xs font-mono text-neutral-400 space-y-1">
              <span className="text-neutral-200 font-semibold block">Key Obstacle:</span>
              <span>Memory bandwidth bottlenecks in multi-token generation on edge accelerators (6GB VRAM).</span>
            </div>
          </div>

          <div className="p-6 rounded-2xl border border-neutral-800 bg-neutral-900/70 space-y-4">
            <div className="flex items-center gap-2 text-amber-400 font-mono text-xs font-bold uppercase tracking-wider">
              <Layers className="w-4 h-4" />
              <span>Existing Approach &amp; Limitations</span>
            </div>
            <p className="text-xs sm:text-sm text-neutral-300 leading-relaxed font-sans">
              {data.existingApproach}
            </p>
            <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950 text-xs font-mono text-neutral-400 space-y-1">
              <span className="text-neutral-200 font-semibold block">Prior Gap:</span>
              <span>No prior study investigates whether post-pruning activation outlier shifts disrupt AWQ scaling factors.</span>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'improvement',
      title: 'Proposed Unified Pipeline & Methodology',
      tag: 'Slide 3 of 6',
      content: (
        <div className="space-y-5">
          <div className="p-6 rounded-2xl border border-cyan-500/30 bg-neutral-900/70 space-y-3">
            <div className="text-cyan-400 font-mono text-xs font-bold uppercase tracking-wider">
              Proposed Improvement
            </div>
            <p className="text-sm text-neutral-200 leading-relaxed font-sans">
              {data.proposedImprovement}
            </p>
          </div>

          <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-950 text-xs font-mono">
            <div className="text-neutral-500 text-[11px] uppercase tracking-wider mb-2 text-center">
              Sequential Execution Pipeline
            </div>
            <div className="flex flex-wrap items-center justify-center gap-2 text-neutral-300">
              <span className="px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800">Original Bunny-3B</span>
              <span className="text-cyan-400">→</span>
              <span className="px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800 text-emerald-400 font-semibold">Taylor Prune (20%)</span>
              <span className="text-cyan-400">→</span>
              <span className="px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800 text-amber-300 font-semibold">LoRA Recovery</span>
              <span className="text-cyan-400">→</span>
              <span className="px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800 text-cyan-300 font-semibold">AWQ INT4 Packing</span>
              <span className="text-cyan-400">→</span>
              <span className="px-2.5 py-1 rounded bg-neutral-900 border border-neutral-800">POPE / ScienceQA Eval</span>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'implementation_status',
      title: 'Implementation State & Hardware Constraints',
      tag: 'Slide 4 of 6',
      content: (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/70 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
              <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-bold">
                <CheckCircle2 className="w-4 h-4" />
                <span>Verified Milestones</span>
              </div>
              <ResearchBadge status="VERIFIED" size="sm" />
            </div>
            <ul className="space-y-2 text-xs text-neutral-300 font-sans">
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✔</span>
                <span>Bunny multimodal environment configured and dependencies linked.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✔</span>
                <span>Phi-2 language backbone loads weights cleanly in memory.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✔</span>
                <span>AWQ INT4 integration and scale search algorithm validated.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="text-emerald-400 mt-0.5">✔</span>
                <span>Structured pruning smoke test generated 20% width-pruned checkpoint.</span>
              </li>
            </ul>
          </div>

          <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/70 space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-neutral-800">
              <div className="flex items-center gap-2 text-rose-400 font-mono text-xs font-bold">
                <AlertTriangle className="w-4 h-4" />
                <span>Hardware Constraints &amp; Limits</span>
              </div>
              <ResearchBadge status="BLOCKED" size="sm" />
            </div>
            <p className="text-xs text-neutral-300 leading-relaxed font-sans">
              {data.hardwareConstraint}
            </p>
            <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950 text-xs font-mono space-y-1">
              <div className="flex justify-between text-neutral-400"><span>Local GPU:</span><span className="text-neutral-200">RTX 3050 (6GB VRAM)</span></div>
              <div className="flex justify-between text-neutral-400"><span>Multimodal Recovery Need:</span><span className="text-rose-400 font-bold">&gt;14GB VRAM</span></div>
              <div className="flex justify-between text-neutral-400"><span>Solution:</span><span className="text-cyan-300">Cloud GPU Allocation</span></div>
            </div>
          </div>
        </div>
      )
    },
    {
      id: 'research_questions',
      title: 'Core Research Questions Under Study',
      tag: 'Slide 5 of 6',
      content: (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {data.researchQuestions.map((item, idx) => (
            <div
              key={idx}
              className="p-6 rounded-2xl border border-neutral-800 bg-neutral-900/70 space-y-3"
            >
              <div className="flex items-center gap-2 text-cyan-400 font-mono text-xs uppercase tracking-wider font-bold">
                <HelpCircle className="w-4 h-4" />
                <span>Research Question {idx + 1}</span>
              </div>
              <h3 className="text-sm font-bold text-neutral-100 leading-snug">
                {item.q}
              </h3>
              <p className="text-xs text-neutral-400 leading-relaxed font-sans">
                {item.context}
              </p>
            </div>
          ))}
        </div>
      )
    },
    {
      id: 'outcomes',
      title: 'Expected Outcomes & Scientific Integrity',
      tag: 'Slide 6 of 6',
      content: (
        <div className="space-y-5">
          <div className="p-6 rounded-2xl border border-emerald-500/30 bg-neutral-900/70 space-y-3">
            <div className="text-emerald-400 font-mono text-xs font-bold uppercase tracking-wider">
              Expected Research Outcomes
            </div>
            <p className="text-sm text-neutral-200 leading-relaxed font-sans">
              {data.expectedOutcome}
            </p>
          </div>

          <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 flex items-start gap-3">
            <ShieldAlert className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
            <div className="space-y-1 text-xs text-neutral-300 font-sans">
              <strong className="text-amber-300 font-mono">Scientific Rigor Commitment:</strong>
              <p className="text-neutral-400 leading-relaxed">
                All benchmark scores, inference latencies, and accuracy deltas remain designated as <span className="text-amber-300 font-mono">PENDING</span> until cloud execution is finalized. No fictitious experimental results are shown to reviewers.
              </p>
            </div>
          </div>
        </div>
      )
    }
  ];

  const currentSlide = slides[slideIndex];

  return (
    <div id="presentation-mode" className="fixed inset-0 z-50 bg-neutral-950 text-neutral-100 flex flex-col justify-between overflow-hidden">
      <div className="flex items-center justify-between px-6 py-4 border-b border-neutral-800 bg-neutral-900/90 backdrop-blur-md">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center text-cyan-400 font-mono font-bold text-xs">
            PR
          </div>
          <div>
            <div className="text-xs font-mono font-bold text-neutral-200 flex items-center gap-2">
              <span>Project Review Deck (Under 2 Minutes)</span>
              <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
                {currentSlide.tag}
              </span>
            </div>
            <div className="text-[11px] text-neutral-400">
              Bunny-v1_0-3B • Structured Pruning • AWQ INT4
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={() => onNavigateView('dashboard')}
            className="px-3 py-1.5 rounded-lg border border-neutral-700 bg-neutral-800 text-xs font-mono text-neutral-300 hover:bg-neutral-700 transition-colors"
          >
            Open Full Dashboard
          </button>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg border border-neutral-700 bg-neutral-800 text-neutral-400 hover:text-neutral-200 hover:bg-neutral-700 transition-colors"
            title="Exit Presentation"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

      <div className="flex-1 max-w-5xl w-full mx-auto p-6 sm:p-10 flex flex-col justify-center overflow-y-auto">
        <div className="text-xs font-mono text-cyan-400 uppercase tracking-widest mb-1">
          {currentSlide.title}
        </div>
        <div className="pt-2">
          {currentSlide.content}
        </div>
      </div>

      <div className="flex items-center justify-between px-6 py-4 border-t border-neutral-800 bg-neutral-900/80">
        <div className="flex items-center gap-1.5">
          {slides.map((_, idx) => (
            <button
              key={idx}
              onClick={() => setSlideIndex(idx)}
              className={`h-2 rounded-full transition-all ${
                slideIndex === idx ? 'w-8 bg-cyan-400' : 'w-2 bg-neutral-700 hover:bg-neutral-600'
              }`}
            />
          ))}
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={() => setSlideIndex(prev => Math.max(0, prev - 1))}
            disabled={slideIndex === 0}
            className={`flex items-center gap-1 px-3.5 py-1.5 rounded-lg border font-mono text-xs transition-colors ${
              slideIndex === 0
                ? 'border-neutral-800 text-neutral-600 cursor-not-allowed'
                : 'border-neutral-700 bg-neutral-800 hover:bg-neutral-700 text-neutral-200'
            }`}
          >
            <ArrowLeft className="w-3.5 h-3.5" />
            <span>Previous</span>
          </button>

          <button
            onClick={() => setSlideIndex(prev => Math.min(slides.length - 1, prev + 1))}
            disabled={slideIndex === slides.length - 1}
            className={`flex items-center gap-1 px-4 py-1.5 rounded-lg border font-mono text-xs font-bold transition-colors ${
              slideIndex === slides.length - 1
                ? 'border-neutral-800 text-neutral-600 cursor-not-allowed'
                : 'bg-cyan-500 hover:bg-cyan-400 text-neutral-950 border-cyan-400 shadow-[0_0_15px_rgba(6,182,212,0.3)]'
            }`}
          >
            <span>Next</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
};
