import React from 'react';
import { ResearchStatus } from '../../types/research';
import { ResearchBadge } from './Badge';

interface StatCardProps {
  id?: string;
  title: string;
  value: string | number | null;
  unit?: string;
  status: ResearchStatus;
  note?: string;
  icon?: React.ReactNode;
  highlight?: boolean;
}

export const StatCard: React.FC<StatCardProps> = ({
  id,
  title,
  value,
  unit,
  status,
  note,
  icon,
  highlight = false
}) => {
  return (
    <div
      id={id || `card-${title.toLowerCase().replace(/[^a-z0-9]/g, '-')}`}
      className={`relative p-4 rounded-xl border transition-all ${
        highlight
          ? 'bg-neutral-900/90 border-cyan-500/40 shadow-[0_4px_24px_rgba(6,182,212,0.08)]'
          : 'bg-neutral-900/60 border-neutral-800 hover:border-neutral-700'
      }`}
    >
      <div className="flex items-start justify-between gap-2 mb-2">
        <div className="flex items-center gap-2">
          {icon && <span className="text-neutral-400">{icon}</span>}
          <span className="text-xs font-medium text-neutral-400 uppercase tracking-wider">{title}</span>
        </div>
        <ResearchBadge status={status} size="sm" />
      </div>

      <div className="flex items-baseline gap-1.5 my-1">
        <span className="text-2xl font-bold font-mono tracking-tight text-neutral-100">
          {value !== null && value !== undefined && value !== '' ? value : 'Pending'}
        </span>
        {unit && <span className="text-xs font-mono text-neutral-400">{unit}</span>}
      </div>

      {note && <div className="text-[11px] text-neutral-400 mt-1 line-clamp-2 leading-relaxed">{note}</div>}
    </div>
  );
};
