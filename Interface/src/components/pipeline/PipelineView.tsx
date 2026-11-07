import React, { useState } from 'react';
import { PipelineStageInfo } from '../../types/research';
import { ResearchBadge, ExecutionBadge } from '../common/Badge';
import {
  GitCommit,
  Eye,
  Cpu,
  Layers,
  Sparkles,
  ArrowDown,
  CheckCircle2,
  AlertOctagon,
  Maximize2,
  Sliders,
  Database
} from 'lucide-react';

interface PipelineViewProps {
  stages: PipelineStageInfo[];
  onSelectStage: (stage: PipelineStageInfo) => void;
}

export const PipelineView: React.FC<PipelineViewProps> = ({ stages, onSelectStage }) => {
  const [selectedNodeKey, setSelectedNodeKey] = useState<string>('pruning');

  const architectureNodes = [
    {
      key: 'vlm_root',
      title: 'Large Vision-Language Model',
      subtitle: 'Multimodal Foundation Architecture',
      spec: 'Bunny-v1_0-3B (~3.12B Total Params)',
      type: 'root',
      icon: <Sparkles className="w-4 h-4 text-cyan-400" />
    },
    {
      key: 'vision_encoder',
      title: 'Vision Encoder',
      subtitle: 'SigLIP-SO400M (Patch-14, 384×384)',
      spec: '428M Parameters • Hidden Dim 1152',
      type: 'component',
      icon: <Eye className="w-4 h-4 text-blue-400" />
    },
    {
      key: 'projector',
      title: 'Projector / Cross-Modal Adapter',
      subtitle: '2-Layer MLP (GELU Activation)',
      spec: 'Maps 1152 (SigLIP) → 2560 (Phi-2)',
      type: 'component',
      icon: <Sliders className="w-4 h-4 text-indigo-400" />
    },
    {
      key: 'language_backbone',
      title: 'Language Backbone',
      subtitle: 'Microsoft Phi-2 (Causal Transformer)',
      spec: '2.78B Parameters • 32 Layers • 32 Heads',
      type: 'component',
      icon: <Cpu className="w-4 h-4 text-purple-400" />
    },
    {
      key: 'pruning',
      title: 'Structured Pruning',
      subtitle: 'Taylor First-Order Channel Pruning',
      spec: 'LLM-Pruner Engine • Width-Wise 20% Cut',
      type: 'optimization',
      icon: <GitCommit className="w-4 h-4 text-emerald-400" />
    },
    {
      key: 'recovery',
      title: 'Recovery Fine-Tuning',
      subtitle: 'Post-Pruning Quality Restoration',
      spec: 'LoRA Adapter Optimization (Cloud Target)',
      type: 'optimization',
      icon: <Layers className="w-4 h-4 text-amber-400" />
    },
    {
      key: 'calibration',
      title: 'Activation Calibration',
      subtitle: 'Forward Outlier Variance Profiling',
      spec: '128 Multimodal Sequences (ShareGPT4V)',
      type: 'optimization',
      icon: <Database className="w-4 h-4 text-cyan-400" />
    },
    {
      key: 'awq_int4',
      title: 'AWQ INT4 Quantization',
      subtitle: 'Activation-Aware Weight Quantization',
      spec: 'W4A16 GEMM Kernels • Group Size 128',
      type: 'optimization',
      icon: <Sparkles className="w-4 h-4 text-cyan-400" />
    },
    {
      key: 'compressed_vlm',
      title: 'Compressed LVLM',
      subtitle: 'Deployable Compact Multimodal Model',
      spec: '~2.56B Params (Packed 4-Bit Weight Storage)',
      type: 'result',
      icon: <CheckCircle2 className="w-4 h-4 text-emerald-400" />
    },
    {
      key: 'evaluation',
      title: 'Benchmark & Evaluation',
      subtitle: 'Hallucination & Accuracy Profiling',
      spec: 'POPE, ScienceQA, MME, Hardware Latency',
      type: 'result',
      icon: <AlertOctagon className="w-4 h-4 text-amber-400" />
    }
  ];

  const nodeDetails: Record<string, {
    description: string;
    mathFormulation?: string;
    rationale: string;
    status: string;
    statusBadge: 'VERIFIED' | 'PENDING' | 'BLOCKED' | 'DEMO';
  }> = {
    vlm_root: {
      description: 'The base multimodal system combines a vision transformer with a pre-trained causal language model through a cross-modal projector.',
      rationale: 'Bunny-3B provides a balanced high-performing testbed that fits within academic research budgets while retaining rich visual instruction capabilities.',
      status: 'Base architecture instantiated and environment verified.',
      statusBadge: 'VERIFIED'
    },
    vision_encoder: {
      description: 'SigLIP (Simple Gated Contrastive Image-Text Pre-training) encodes 384x384 image patches into continuous visual tokens.',
      rationale: 'Left frozen or unpruned in initial phase to protect primary visual feature representations from degradation.',
      status: 'SigLIP SO400M weights loaded successfully without shape mismatch.',
      statusBadge: 'VERIFIED'
    },
    projector: {
      description: 'A two-layer projection MLP with GELU non-linearity projecting image representations from dimension 1152 to language space 2560.',
      rationale: 'Bridges modality misalignment; essential for instruction tuning recovery.',
      status: 'Projector layer weights loaded and verified in memory.',
      statusBadge: 'VERIFIED'
    },
    language_backbone: {
      description: 'Microsoft Phi-2 architecture composed of 32 transformer layers, 2560 hidden states, and 10240 intermediate MLP dimensions.',
      rationale: 'Represents ~90% of model parameter volume and is the primary candidate for aggressive structured width reduction.',
      status: 'Phi-2 language backbone loaded successfully.',
      statusBadge: 'VERIFIED'
    },
    pruning: {
      description: 'Structured width pruning removes entire coupled channels across Query/Key/Value and MLP projections using Taylor first-order gradient sensitivity.',
      mathFormulation: 'I(W) = |∂L/∂W ⊙ W| (Taylor First-Order Expansion)',
      rationale: 'Unlike unstructured sparsity which requires specialized sparse GEMM accelerators, structured channel removal directly yields physical speedups on standard dense matrix hardware.',
      status: 'Pruning smoke test successfully generated a 20% width-pruned checkpoint.',
      statusBadge: 'VERIFIED'
    },
    recovery: {
      description: 'Lightweight recovery training using Low-Rank Adaptation (LoRA) on attention projection weights to recover lost linguistic and cross-modal capabilities.',
      mathFormulation: 'W_recovered = W_pruned + ΔW, where ΔW = B · A (Rank r=16)',
      rationale: 'Direct zero-shot evaluation immediately post-pruning shows significant score drops; recovery restores alignment.',
      status: 'Blocked on local hardware (RTX 3050 6GB VRAM limit); designated for Cloud GPU execution.',
      statusBadge: 'BLOCKED'
    },
    calibration: {
      description: 'Collects forward pass activation statistics using multimodal prompt-image pairs to record channel outlier behavior.',
      mathFormulation: 's_X = E[|X|] per channel across calibration distribution',
      rationale: 'Identifies which channels carry disproportionate activation energy and therefore must be shielded from quantization noise.',
      status: 'Calibration hook implementation verified on language backbone.',
      statusBadge: 'VERIFIED'
    },
    awq_int4: {
      description: 'Activation-Aware Weight Quantization searches for optimal per-channel scaling factors s that minimize quantization error on salient channels before rounding.',
      mathFormulation: 'min_s || W · X - Q(W · diag(s)⁻¹) · diag(s) · X ||²',
      rationale: 'Quantizes weights to 4-bit while maintaining 16-bit FP16 activations (W4A16), slashing memory traffic by ~3.8x.',
      status: 'AWQ W4A16 layer replacement verified in smoke test.',
      statusBadge: 'VERIFIED'
    },
    compressed_vlm: {
      description: 'The unified model combining 20% physical channel reduction with INT4 packed weights, reducing parameter overhead substantially.',
      rationale: 'Target artifact for edge vision-language deployment on resource-constrained consumer GPUs.',
      status: 'Awaiting completion of cloud-based recovery before final packaging.',
      statusBadge: 'PENDING'
    },
    evaluation: {
      description: 'Downstream scientific evaluation across POPE (hallucination), ScienceQA (multimodal reasoning), and hardware latency/VRAM benchmarking.',
      rationale: 'Determines the Pareto efficiency boundary between uncompressed, INT8, and AWQ INT4 pruned models.',
      status: 'Evaluation harness scripts configured; execution marked pending until model pipeline finishes.',
      statusBadge: 'PENDING'
    }
  };

  const activeDetail = nodeDetails[selectedNodeKey] || nodeDetails['pruning'];

  return (
    <div id="pipeline-view" className="space-y-6">
      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider">
              System Architecture
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-300 font-mono">
              College Review Architecture Diagram
            </span>
          </div>
          <h2 className="text-lg font-bold text-neutral-100">End-to-End VLM Compression Pipeline</h2>
          <p className="text-xs text-neutral-400 mt-0.5 max-w-2xl">
            From the original Bunny-v1_0-3B foundation model to structurally pruned and AWQ INT4 quantized edge artifacts.
          </p>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <span className="text-xs font-mono text-neutral-400">Click any block to inspect details</span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 space-y-3">
          <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-950/70 shadow-2xl">
            <div className="text-xs font-mono text-neutral-500 uppercase tracking-widest text-center mb-4">
              ─── RESEARCH PIPELINE DATAFLOW ───
            </div>

            <div className="space-y-2">
              {architectureNodes.map((node, index) => {
                const isSelected = selectedNodeKey === node.key;
                return (
                  <React.Fragment key={node.key}>
                    <button
                      onClick={() => setSelectedNodeKey(node.key)}
                      className={`w-full text-left p-3.5 rounded-xl border transition-all flex items-center justify-between group ${
                        isSelected
                          ? 'bg-neutral-800/90 border-cyan-500/60 shadow-[0_0_20px_rgba(6,182,212,0.18)] translate-x-1'
                          : 'bg-neutral-900/60 border-neutral-800 hover:border-neutral-700 hover:bg-neutral-900'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className={`w-8 h-8 rounded-lg flex items-center justify-center border ${
                            isSelected
                              ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300'
                              : 'bg-neutral-800/80 border-neutral-700 text-neutral-400'
                          }`}
                        >
                          {node.icon}
                        </div>
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-bold text-neutral-100">{node.title}</span>
                            <span className="text-[10px] font-mono text-neutral-500">({node.subtitle})</span>
                          </div>
                          <div className="text-[11px] font-mono text-neutral-400 mt-0.5">{node.spec}</div>
                        </div>
                      </div>

                      <div className="shrink-0 flex items-center gap-2">
                        <ResearchBadge status={nodeDetails[node.key]?.statusBadge || 'VERIFIED'} size="sm" />
                      </div>
                    </button>

                    {index < architectureNodes.length - 1 && (
                      <div className="flex justify-center my-0.5">
                        <div className="w-0.5 h-3 bg-neutral-800 flex items-center justify-center">
                          <ArrowDown className="w-3 h-3 text-neutral-600" />
                        </div>
                      </div>
                    )}
                  </React.Fragment>
                );
              })}
            </div>
          </div>
        </div>

        <div className="lg:col-span-5 space-y-4">
          <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/60 sticky top-20 space-y-4">
            <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
              <div>
                <span className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider font-semibold block">
                  Node Inspector
                </span>
                <h3 className="text-sm font-bold text-neutral-100">
                  {architectureNodes.find(n => n.key === selectedNodeKey)?.title}
                </h3>
              </div>
              <ResearchBadge status={activeDetail.statusBadge} />
            </div>

            <div className="space-y-3 font-mono text-xs">
              <div>
                <span className="text-neutral-500 text-[11px] uppercase tracking-wider block mb-1">
                  Architectural Description
                </span>
                <p className="font-sans text-xs text-neutral-300 leading-relaxed bg-neutral-950/60 p-3 rounded-xl border border-neutral-800/80">
                  {activeDetail.description}
                </p>
              </div>

              {activeDetail.mathFormulation && (
                <div>
                  <span className="text-neutral-500 text-[11px] uppercase tracking-wider block mb-1">
                    Mathematical Formulation
                  </span>
                  <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950 font-mono text-xs text-cyan-300 text-center">
                    {activeDetail.mathFormulation}
                  </div>
                </div>
              )}

              <div>
                <span className="text-neutral-500 text-[11px] uppercase tracking-wider block mb-1">
                  Scientific Rationale
                </span>
                <p className="font-sans text-xs text-neutral-300 leading-relaxed bg-neutral-950/60 p-3 rounded-xl border border-neutral-800/80">
                  {activeDetail.rationale}
                </p>
              </div>

              <div>
                <span className="text-neutral-500 text-[11px] uppercase tracking-wider block mb-1">
                  Current Verification State
                </span>
                <div className="p-3 rounded-xl border border-neutral-800 bg-neutral-950/80 flex items-start gap-2">
                  <div className="mt-0.5">
                    {activeDetail.statusBadge === 'VERIFIED' && <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />}
                    {activeDetail.statusBadge === 'BLOCKED' && <AlertOctagon className="w-3.5 h-3.5 text-rose-400" />}
                    {activeDetail.statusBadge === 'PENDING' && <AlertOctagon className="w-3.5 h-3.5 text-amber-400" />}
                  </div>
                  <span className="text-xs text-neutral-300">{activeDetail.status}</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
        <div className="flex items-center gap-2 pb-3 border-b border-neutral-800">
          <GitCommit className="w-4 h-4 text-emerald-400" />
          <h3 className="text-sm font-bold text-neutral-100">Structured Pruning Mechanics Explained</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-950/60 space-y-2">
            <div className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider font-semibold">
              Step 1: Baseline Width
            </div>
            <h4 className="text-xs font-bold text-neutral-200">Dense Channel Topology</h4>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Standard Bunny-3B language layers feature 10240 intermediate MLP dimensions and 2560 hidden states across 32 transformer blocks.
            </p>
            <div className="pt-2 font-mono text-[11px] text-neutral-500">
              Shape: [32, 2560, 10240]
            </div>
          </div>

          <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-950/60 space-y-2">
            <div className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider font-semibold">
              Step 2: Taylor Sensitivity
            </div>
            <h4 className="text-xs font-bold text-neutral-200">First-Order Gradient Saliency</h4>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Channels are evaluated on prompt batches. The bottom 20% of channels contributing least to cross-entropy loss are targeted for removal.
            </p>
            <div className="pt-2 font-mono text-[11px] text-emerald-400">
              Coupled Dependency Preserved
            </div>
          </div>

          <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-950/60 space-y-2">
            <div className="text-[10px] font-mono text-cyan-400 uppercase tracking-wider font-semibold">
              Step 3: Compact Architecture
            </div>
            <h4 className="text-xs font-bold text-neutral-200">Physical Channel Slicing</h4>
            <p className="text-xs text-neutral-400 leading-relaxed">
              Intermediate dimensions are physically shrunk from 10240 to 8192, directly cutting memory bandwidth and floating point operations.
            </p>
            <div className="pt-2 font-mono text-[11px] text-cyan-300">
              Pruned Checkpoint Generated
            </div>
          </div>
        </div>

        <div className="p-3 rounded-xl border border-amber-500/20 bg-amber-500/5 text-xs text-neutral-300 flex items-start gap-2.5">
          <AlertOctagon className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-0.5">
            <strong className="text-amber-300 font-semibold">Scientific Rigor Notice:</strong>
            <p className="text-[11px] text-neutral-400 leading-relaxed">
              Structured pruning reduces memory consumption and parameter count, but does NOT automatically guarantee accuracy retention. Downstream multimodal benchmarks (POPE, ScienceQA) are required to evaluate empirical fidelity after recovery.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
