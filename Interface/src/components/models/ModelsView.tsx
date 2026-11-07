import React from 'react';
import { BUNNY_MODEL_SPEC } from '../../data/researchData';
import { ResearchBadge } from '../common/Badge';
import {
  Box,
  Eye,
  Cpu,
  Layers,
  Database,
  CheckCircle2,
  FileCode,
  ShieldCheck,
  Hash
} from 'lucide-react';

export const ModelsView: React.FC = () => {
  const model = BUNNY_MODEL_SPEC;

  const layerBreakdown = [
    { module: 'Vision Encoder: Patch Embed', type: 'Conv2d(3, 1152, kernel=14, stride=14)', params: '0.68M', precision: 'FP16', status: 'VERIFIED' },
    { module: 'Vision Encoder: SigLIP Blocks (x27)', type: 'TransformerEncoderLayer(dim=1152, heads=16)', params: '427.3M', precision: 'FP16', status: 'VERIFIED' },
    { module: 'Cross-Modal Projector: Layer 1', type: 'nn.Linear(1152, 2560) + GELU', params: '2.95M', precision: 'FP16', status: 'VERIFIED' },
    { module: 'Cross-Modal Projector: Layer 2', type: 'nn.Linear(2560, 2560)', params: '6.55M', precision: 'FP16', status: 'VERIFIED' },
    { module: 'Language Backbone: Embeddings', type: 'nn.Embedding(51200, 2560)', params: '131.07M', precision: 'FP16', status: 'VERIFIED' },
    { module: 'Language Backbone: Attention (q,k,v,out) x32', type: 'CausalSelfAttention (dim=2560, heads=32)', params: '838.86M', precision: 'FP16 / INT4 Target', status: 'VERIFIED' },
    { module: 'Language Backbone: MLP (fc1, fc2) x32', type: 'MLP (2560 → 10240 → 2560)', params: '1,677.72M', precision: 'FP16 / INT4 Target', status: 'VERIFIED' },
    { module: 'Language Backbone: LM Head', type: 'nn.Linear(2560, 51200, bias=False)', params: '131.07M', precision: 'FP16 (Preserved)', status: 'VERIFIED' }
  ];

  return (
    <div id="models-view" className="space-y-6">
      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/60 flex flex-col md:flex-row md:items-center justify-between gap-4">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <span className="text-xs font-mono font-semibold text-cyan-400 uppercase tracking-wider">
              Architecture Inventory
            </span>
            <span className="text-[10px] px-2 py-0.5 rounded bg-emerald-950/60 border border-emerald-800/60 text-emerald-300 font-mono">
              Weights Verified
            </span>
          </div>
          <h2 className="text-lg font-bold text-neutral-100">{model.name} Multimodal Specification</h2>
          <p className="text-xs text-neutral-400 mt-0.5">
            Detailed layer structural breakdown, tensor dimensional mappings, and target modules for compression.
          </p>
        </div>

        <div className="flex items-center gap-2 font-mono text-xs text-neutral-400 bg-neutral-950 px-3 py-1.5 rounded-xl border border-neutral-800">
          <span>Total Parameters:</span>
          <span className="text-cyan-300 font-bold">{model.totalParameters}</span>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/50 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-neutral-200 text-xs font-bold">
              <Eye className="w-4 h-4 text-blue-400" />
              <span>Vision Encoder</span>
            </div>
            <ResearchBadge status="VERIFIED" size="sm" />
          </div>
          <div className="font-mono text-xs space-y-1.5">
            <div className="text-neutral-200 font-semibold">{model.visionEncoder.name}</div>
            <div className="text-neutral-400 text-[11px]">{model.visionEncoder.architecture}</div>
            <div className="pt-2 border-t border-neutral-800/80 space-y-1 text-[11px] text-neutral-500">
              <div className="flex justify-between"><span>Parameters:</span><span className="text-neutral-300">{model.visionEncoder.parameters}</span></div>
              <div className="flex justify-between"><span>Resolution:</span><span className="text-neutral-300">{model.visionEncoder.resolution}</span></div>
              <div className="flex justify-between"><span>Patch Size:</span><span className="text-neutral-300">{model.visionEncoder.patchSize} px</span></div>
              <div className="flex justify-between"><span>Hidden Dim:</span><span className="text-neutral-300">{model.visionEncoder.hiddenDimension}</span></div>
            </div>
          </div>
        </div>

        <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/50 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-neutral-200 text-xs font-bold">
              <Layers className="w-4 h-4 text-indigo-400" />
              <span>Cross-Modal Projector</span>
            </div>
            <ResearchBadge status="VERIFIED" size="sm" />
          </div>
          <div className="font-mono text-xs space-y-1.5">
            <div className="text-neutral-200 font-semibold">{model.projector.type}</div>
            <div className="text-neutral-400 text-[11px]">Feature projection adapter</div>
            <div className="pt-2 border-t border-neutral-800/80 space-y-1 text-[11px] text-neutral-500">
              <div className="flex justify-between"><span>Layers:</span><span className="text-neutral-300">{model.projector.layers} linear dense</span></div>
              <div className="flex justify-between"><span>Input Dimension:</span><span className="text-neutral-300">{model.projector.inputDim} (Vision)</span></div>
              <div className="flex justify-between"><span>Output Dimension:</span><span className="text-neutral-300">{model.projector.outputDim} (Language)</span></div>
              <div className="flex justify-between"><span>Total Weights:</span><span className="text-neutral-300">~9.5M parameters</span></div>
            </div>
          </div>
        </div>

        <div className="p-4 rounded-xl border border-neutral-800 bg-neutral-900/50 space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-neutral-200 text-xs font-bold">
              <Cpu className="w-4 h-4 text-purple-400" />
              <span>Language Backbone</span>
            </div>
            <ResearchBadge status="VERIFIED" size="sm" />
          </div>
          <div className="font-mono text-xs space-y-1.5">
            <div className="text-neutral-200 font-semibold">{model.languageBackbone.name}</div>
            <div className="text-neutral-400 text-[11px]">{model.languageBackbone.architecture}</div>
            <div className="pt-2 border-t border-neutral-800/80 space-y-1 text-[11px] text-neutral-500">
              <div className="flex justify-between"><span>Parameters:</span><span className="text-neutral-300">{model.languageBackbone.parameters}</span></div>
              <div className="flex justify-between"><span>Layers:</span><span className="text-neutral-300">{model.languageBackbone.layers} Transformer blocks</span></div>
              <div className="flex justify-between"><span>Intermediate Dim:</span><span className="text-neutral-300">{model.languageBackbone.intermediateDimension} (MLP)</span></div>
              <div className="flex justify-between"><span>Attention Heads:</span><span className="text-neutral-300">{model.languageBackbone.attentionHeads}</span></div>
            </div>
          </div>
        </div>
      </div>

      <div className="p-5 rounded-2xl border border-neutral-800 bg-neutral-900/50 space-y-4">
        <div className="flex items-center justify-between pb-3 border-b border-neutral-800">
          <div>
            <h3 className="text-sm font-bold text-neutral-100 flex items-center gap-2">
              <Hash className="w-4 h-4 text-cyan-400" />
              <span>Layer Structural Breakdown &amp; Pruning Target Map</span>
            </h3>
            <p className="text-xs text-neutral-400 mt-0.5">
              Identifies which weight tensors undergo Taylor width pruning and subsequent AWQ INT4 quantization.
            </p>
          </div>
          <span className="text-xs font-mono text-neutral-500">8 Structural Groups</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-neutral-800 text-neutral-400 text-[11px]">
                <th className="py-2.5 px-3">Module Component</th>
                <th className="py-2.5 px-3">Tensor Architecture</th>
                <th className="py-2.5 px-3">Parameters</th>
                <th className="py-2.5 px-3">Target Precision</th>
                <th className="py-2.5 px-3 text-right">Verification</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-850">
              {layerBreakdown.map((row, idx) => (
                <tr key={idx} className="hover:bg-neutral-800/40 transition-colors">
                  <td className="py-2.5 px-3 font-semibold text-neutral-200">{row.module}</td>
                  <td className="py-2.5 px-3 text-neutral-400">{row.type}</td>
                  <td className="py-2.5 px-3 text-cyan-300 font-bold">{row.params}</td>
                  <td className="py-2.5 px-3 text-neutral-300">{row.precision}</td>
                  <td className="py-2.5 px-3 text-right">
                    <ResearchBadge status="VERIFIED" size="sm" />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
