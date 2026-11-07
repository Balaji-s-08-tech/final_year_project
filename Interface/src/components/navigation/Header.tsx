import React from 'react';
import { ExecutionEnvironment } from '../../types/research';
import { Cpu, Radio, ShieldCheck, PlayCircle, Layers, Server } from 'lucide-react';

interface HeaderProps {
  currentExperimentId: string;
  executionMode: ExecutionEnvironment;
  onSelectExecutionMode: (mode: ExecutionEnvironment) => void;
  isMockMode: boolean;
  onToggleMockMode: (mock: boolean) => void;
  onOpenPresentation: () => void;
  isPresentationActive: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentExperimentId,
  executionMode,
  onSelectExecutionMode,
  isMockMode,
  onToggleMockMode,
  onOpenPresentation,
  isPresentationActive
}) => {
  return (
    <header id="app-header" className="sticky top-0 z-30 border-b border-neutral-800 bg-neutral-950/95 backdrop-blur-md px-4 sm:px-6 py-3">
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500/20 to-neutral-900 border border-cyan-500/30 flex items-center justify-center text-cyan-400 font-mono font-bold text-sm shadow-[0_0_15px_rgba(6,182,212,0.15)]">
            VLM
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-sm sm:text-base font-bold text-neutral-100 tracking-tight leading-none">
                LVLM Compression Lab
              </h1>
              <span className="text-[10px] px-2 py-0.5 rounded bg-cyan-950/60 border border-cyan-800/60 text-cyan-300 font-mono">
                Bunny-v1_0-3B
              </span>
            </div>
            <p className="text-[11px] text-neutral-400 line-clamp-1 mt-0.5">
              Enhancing Structured Pruning with Activation-Aware Weight Quantization (AWQ)
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-2.5 sm:gap-3 text-xs">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-neutral-800 bg-neutral-900/80 font-mono text-[11px]">
            <span className="text-neutral-500">Exp:</span>
            <span className="text-cyan-300 font-semibold">{currentExperimentId}</span>
          </div>

          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg border border-neutral-800 bg-neutral-900/80 text-[11px] font-mono">
            <Radio className="w-3 h-3 text-emerald-400 animate-pulse" />
            <span className="text-neutral-400">System:</span>
            <span className="text-emerald-400 font-medium">Ready</span>
          </div>

          <button
            id="btn-backend-status"
            onClick={() => onToggleMockMode(!isMockMode)}
            className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] font-mono transition-all ${
              isMockMode
                ? 'bg-amber-500/10 border-amber-500/30 text-amber-300 hover:bg-amber-500/20'
                : 'bg-emerald-500/10 border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/20'
            }`}
            title="Click to toggle between Mock Simulation and Live Python Backend hook"
          >
            <Server className="w-3 h-3" />
            <span>Backend:</span>
            <span className="font-bold">{isMockMode ? 'DEMO MOCK' : 'LIVE REST'}</span>
          </button>

          <div className="flex items-center rounded-lg border border-neutral-800 bg-neutral-900/80 p-0.5 text-[11px] font-mono">
            <span className="px-2 text-neutral-500 flex items-center gap-1">
              <Cpu className="w-3 h-3 text-neutral-400" />
              <span>Target:</span>
            </span>
            {(['LOCAL', 'CLOUD', 'HYBRID'] as ExecutionEnvironment[]).map(mode => (
              <button
                key={mode}
                onClick={() => onSelectExecutionMode(mode)}
                className={`px-2 py-0.5 rounded transition-all ${
                  executionMode === mode
                    ? mode === 'HYBRID'
                      ? 'bg-cyan-500/20 text-cyan-300 font-semibold border border-cyan-500/40 shadow-sm'
                      : 'bg-neutral-800 text-neutral-200 font-medium'
                    : 'text-neutral-400 hover:text-neutral-300'
                }`}
              >
                {mode}
                {mode === 'HYBRID' && <span className="text-[9px] ml-1 text-cyan-400 font-normal">★</span>}
              </button>
            ))}
          </div>

          <button
            id="btn-presentation-mode"
            onClick={onOpenPresentation}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border font-mono text-xs font-semibold transition-all ${
              isPresentationActive
                ? 'bg-cyan-500 text-black border-cyan-400 shadow-[0_0_20px_rgba(6,182,212,0.4)]'
                : 'bg-gradient-to-r from-neutral-900 to-neutral-800 border-cyan-500/40 text-cyan-300 hover:border-cyan-400 hover:text-cyan-200'
            }`}
          >
            <PlayCircle className="w-3.5 h-3.5" />
            <span>Presentation Mode</span>
          </button>
        </div>
      </div>
    </header>
  );
};
