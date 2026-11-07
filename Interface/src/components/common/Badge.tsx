import React from 'react';
import { ResearchStatus, StageExecutionStatus } from '../../types/research';

interface ResearchBadgeProps {
  status: ResearchStatus;
  size?: 'sm' | 'md' | 'lg';
}

export const ResearchBadge: React.FC<ResearchBadgeProps> = ({ status, size = 'md' }) => {
  const sizeClasses = {
    sm: 'text-[10px] px-1.5 py-0.5 tracking-wider',
    md: 'text-xs px-2.5 py-1 tracking-wider',
    lg: 'text-sm px-3 py-1.5 font-semibold tracking-wider'
  };

  const statusStyles: Record<ResearchStatus, string> = {
    VERIFIED: 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30 shadow-[0_0_12px_rgba(16,185,129,0.15)]',
    PENDING: 'bg-amber-500/10 text-amber-300 border border-amber-500/30',
    BLOCKED: 'bg-rose-500/10 text-rose-300 border border-rose-500/30',
    DEMO: 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/30 shadow-[0_0_12px_rgba(6,182,212,0.15)]'
  };

  const statusIcons: Record<ResearchStatus, string> = {
    VERIFIED: '●',
    PENDING: '◌',
    BLOCKED: '■',
    DEMO: '▲'
  };

  return (
    <span
      id={`badge-${status.toLowerCase()}`}
      className={`inline-flex items-center gap-1.5 rounded font-mono font-medium uppercase whitespace-nowrap ${sizeClasses[size]} ${statusStyles[status]}`}
    >
      <span className="text-[9px] opacity-80">{statusIcons[status]}</span>
      {status}
    </span>
  );
};

interface ExecutionBadgeProps {
  status: StageExecutionStatus;
}

export const ExecutionBadge: React.FC<ExecutionBadgeProps> = ({ status }) => {
  const styles: Record<StageExecutionStatus, string> = {
    idle: 'bg-neutral-800 text-neutral-400 border border-neutral-700',
    pending: 'bg-amber-500/10 text-amber-300 border border-amber-500/30',
    running: 'bg-cyan-500/10 text-cyan-300 border border-cyan-500/40 animate-pulse',
    completed: 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/30',
    failed: 'bg-rose-500/10 text-rose-400 border border-rose-500/30',
    blocked: 'bg-rose-950/40 text-rose-400 border border-rose-800/40'
  };

  return (
    <span
      id={`status-${status}`}
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded text-[11px] font-mono capitalize tracking-wide ${styles[status]}`}
    >
      {status === 'running' && <span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />}
      {status}
    </span>
  );
};
