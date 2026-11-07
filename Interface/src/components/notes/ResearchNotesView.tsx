import React from 'react';
import { FileText, BookOpen, Sparkles, CheckCircle2, AlertOctagon, HelpCircle } from 'lucide-react';
import { ResearchBadge } from '../common/Badge';

export const ResearchNotesView: React.FC = () => {
  return (
    <div id="notes-view" className="space-y-6">
      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider">
              Theoretical Foundation
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 font-mono">
              Academic Documentation
            </span>
          </div>
          <h2 className="text-lg font-bold text-neutral-100">Research Notes, Formulations &amp; Hypotheses</h2>
          <p className="text-xs text-neutral-400 mt-0.5 max-w-2xl">
            Mathematical grounding, multimodal pruning dynamics, and academic references supporting this investigation.
          </p>
        </div>

        <ResearchBadge status="VERIFIED" />
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-3">
          <div className="flex items-center gap-2 text-neutral-200 text-xs font-bold pb-2 border-b border-neutral-800">
            <HelpCircle className="w-4 h-4 text-cyan-400" />
            <span>Research Motivation &amp; Problem Statement</span>
          </div>
          <p className="text-xs text-neutral-300 leading-relaxed">
            While structured pruning provides architectural reduction by physically removing redundant channels, post-pruning models still occupy 16-bit floating point precision. Conversely, weight quantization achieves 4-bit compression but suffers when activation outliers are truncated. By applying Activation-Aware Weight Quantization (AWQ) directly onto a structurally pruned and LoRA-recovered LVLM, we investigate whether the activation outlier landscape shifts after structural channel elimination.
          </p>
        </div>

        <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-3">
          <div className="flex items-center gap-2 text-neutral-200 text-xs font-bold pb-2 border-b border-neutral-800">
            <Sparkles className="w-4 h-4 text-emerald-400" />
            <span>Why Prune Language Backbone Rather than Vision Encoder?</span>
          </div>
          <p className="text-xs text-neutral-300 leading-relaxed">
            In Bunny-v1_0-3B, the SigLIP-SO400M vision encoder accounts for ~428M parameters (13.7%), while Microsoft Phi-2 comprises ~2.78B parameters (86.3%). Vision encoders encode high-entropy continuous signals where spatial tokens are highly sensitive to channel pruning. In contrast, the causal language backbone contains substantial cross-layer parametric redundancy, making width-wise Taylor pruning significantly safer and higher yield.
          </p>
        </div>
      </div>

      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
        <div className="flex items-center gap-2 pb-3 border-b border-neutral-800">
          <BookOpen className="w-4 h-4 text-cyan-400" />
          <h3 className="text-sm font-bold text-neutral-100">Foundational Literature &amp; Citations</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950 space-y-1">
            <div className="text-neutral-200 font-bold">1. Structural Pruning of Large Vision Language Models</div>
            <div className="text-neutral-500 text-[11px]">Comprehensive Study on Pruning Dynamics, Recovery, and Data Efficiency</div>
            <p className="font-sans text-[11px] text-neutral-400 pt-1 leading-normal">
              Establishes the dependency grouping and first-order Taylor criteria for channel pruning in multimodal architectures.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950 space-y-1">
            <div className="text-neutral-200 font-bold">2. AWQ: Activation-aware Weight Quantization for LLM Compression</div>
            <div className="text-neutral-500 text-[11px]">Lin et al. (MIT / NVIDIA, MLSys 2024)</div>
            <p className="font-sans text-[11px] text-neutral-400 pt-1 leading-normal">
              Demonstrates that protecting the top 1% salient weights via per-channel activation scaling prevents quantization perplexity collapse in 4-bit regimes.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950 space-y-1">
            <div className="text-neutral-200 font-bold">3. LLM-Pruner: On the Structural Pruning of Large Language Models</div>
            <div className="text-neutral-500 text-[11px]">Ma et al. (NeurIPS 2023)</div>
            <p className="font-sans text-[11px] text-neutral-400 pt-1 leading-normal">
              Pioneers task-agnostic structural channel extraction with LoRA recovery on pre-trained causal transformers.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-neutral-800 bg-neutral-950 space-y-1">
            <div className="text-neutral-200 font-bold">4. Bunny: A Family of Lightweight Multimodal Models</div>
            <div className="text-neutral-500 text-[11px]">BAAI &amp; Beihang University (2024)</div>
            <p className="font-sans text-[11px] text-neutral-400 pt-1 leading-normal">
              Source model architecture combining SigLIP-SO400M with Phi-2, delivering competitive multimodal reasoning at edge scale.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
