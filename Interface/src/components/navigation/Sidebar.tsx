import React from 'react';
import {
  LayoutDashboard,
  FlaskConical,
  GitFork,
  Box,
  Binary,
  BarChart3,
  TableProperties,
  HardDrive,
  FileText,
  Presentation,
  ShieldAlert
} from 'lucide-react';

export type NavView =
  | 'dashboard'
  | 'experiments'
  | 'pipeline'
  | 'models'
  | 'quantization'
  | 'benchmarks'
  | 'results'
  | 'system'
  | 'notes'
  | 'presentation';

interface SidebarProps {
  currentView: NavView;
  onSelectView: (view: NavView) => void;
  isBackendUnavailable: boolean;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentView, onSelectView, isBackendUnavailable }) => {
  const navItems: { id: NavView; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: 'dashboard', label: 'Dashboard', icon: <LayoutDashboard className="w-4 h-4" /> },
    { id: 'experiments', label: 'Jobs', icon: <FlaskConical className="w-4 h-4" /> },
    { id: 'pipeline', label: 'Pipeline', icon: <GitFork className="w-4 h-4" /> },
    { id: 'models', label: 'Models', icon: <Box className="w-4 h-4" /> },
    { id: 'quantization', label: 'Quantization', icon: <Binary className="w-4 h-4" />, badge: 'AWQ' },
    { id: 'benchmarks', label: 'Benchmarks', icon: <BarChart3 className="w-4 h-4" /> },
    { id: 'results', label: 'Results', icon: <TableProperties className="w-4 h-4" /> },
    { id: 'system', label: 'System & Hardware', icon: <HardDrive className="w-4 h-4" /> }
  ];

  return (
    <aside
      id="app-sidebar"
      className="w-full lg:w-64 border-r border-neutral-800 bg-neutral-950/80 backdrop-blur-sm flex flex-col justify-between shrink-0"
    >
      <div className="p-3 space-y-1">
        <div className="px-3 py-2 text-[10px] font-mono uppercase tracking-widest text-neutral-500 font-semibold">
          Research Console
        </div>
        {navItems.map(item => {
          const isActive = currentView === item.id;
          return (
            <button
              key={item.id}
              id={`nav-${item.id}`}
              onClick={() => onSelectView(item.id)}
              className={`w-full flex items-center justify-between px-3 py-2.5 rounded-lg text-xs font-medium transition-all ${
                isActive
                  ? 'bg-neutral-900 text-cyan-300 border border-neutral-700/80 shadow-sm'
                  : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-900/40'
              }`}
            >
              <div className="flex items-center gap-3">
                <span className={isActive ? 'text-cyan-400' : 'text-neutral-500'}>{item.icon}</span>
                <span>{item.label}</span>
              </div>
              {item.badge && (
                <span
                  className={`text-[10px] font-mono px-1.5 py-0.2 rounded border ${
                    item.id === 'presentation'
                      ? 'bg-cyan-500/10 text-cyan-300 border-cyan-500/30 font-semibold'
                      : 'bg-neutral-800 text-neutral-400 border-neutral-700'
                  }`}
                >
                  {item.badge}
                </span>
              )}
            </button>
          );
        })}
      </div>

      <div className="p-3 m-3 rounded-xl border border-neutral-800/90 bg-neutral-900/40 text-xs">
        <div className="flex items-center gap-2 text-neutral-300 font-mono text-[11px] mb-1.5">
          <ShieldAlert className="w-3.5 h-3.5 text-cyan-400" />
          <span className="font-semibold">Research Integrity</span>
        </div>
        <p className="text-[11px] text-neutral-400 leading-relaxed">
          Unmeasured metrics are explicitly labeled <span className="text-amber-400 font-mono">PENDING</span>. No fabricated results.
        </p>
        <div className="mt-2.5 pt-2 border-t border-neutral-800/80 flex items-center justify-between text-[10px] font-mono text-neutral-500">
          <span>Mode:</span>
          <span className={isBackendUnavailable ? 'text-amber-400 font-semibold' : 'text-emerald-400 font-semibold'}>
            {isBackendUnavailable ? 'BACKEND DISCONNECTED' : 'LIVE API'}
          </span>
        </div>
      </div>
    </aside>
  );
};
