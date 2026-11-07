import React, { useState, useEffect, useRef } from 'react';
import { LogEntry } from '../../types/research';
import { Terminal, Copy, Trash2, Check, ArrowDownCircle, Search, Filter } from 'lucide-react';

interface TerminalViewerProps {
  logs: LogEntry[];
  title?: string;
  maxHeight?: string;
  onClear?: () => void;
}

export const TerminalViewer: React.FC<TerminalViewerProps> = ({
  logs,
  title = 'Experiment Execution Log & Telemetry',
  maxHeight = 'max-h-[380px]',
  onClear
}) => {
  const [filterLevel, setFilterLevel] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const [copied, setCopied] = useState<boolean>(false);
  const logContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (autoScroll && logContainerRef.current) {
      logContainerRef.current.scrollTop = logContainerRef.current.scrollHeight;
    }
  }, [logs, autoScroll]);

  const filteredLogs = logs.filter(log => {
    const matchesLevel = filterLevel === 'ALL' || log.level === filterLevel;
    const matchesSearch =
      searchQuery === '' ||
      log.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      log.stage.toLowerCase().includes(searchQuery.toLowerCase());
    return matchesLevel && matchesSearch;
  });

  const handleCopy = () => {
    const text = filteredLogs
      .map(l => `[${l.timestamp}] [${l.level}] [${l.stage}] ${l.message}`)
      .join('\n');
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'SUCCESS':
        return 'text-emerald-400 font-semibold';
      case 'WARNING':
        return 'text-amber-300 font-semibold';
      case 'ERROR':
        return 'text-rose-400 font-semibold';
      default:
        return 'text-cyan-400';
    }
  };

  return (
    <div id="terminal-viewer" className="rounded-xl border border-neutral-800 bg-neutral-950/90 shadow-2xl overflow-hidden flex flex-col">
      <div className="flex flex-wrap items-center justify-between px-4 py-2.5 bg-neutral-900/80 border-b border-neutral-800 text-xs font-mono">
        <div className="flex items-center gap-2.5 text-neutral-300">
          <Terminal className="w-4 h-4 text-cyan-400" />
          <span className="font-semibold text-neutral-200">{title}</span>
          <span className="text-[10px] px-2 py-0.5 rounded bg-neutral-800 text-neutral-400">
            {filteredLogs.length} events
          </span>
        </div>

        <div className="flex items-center gap-2 mt-2 sm:mt-0">
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-2 text-neutral-500" />
            <input
              type="text"
              placeholder="Filter logs..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              className="pl-8 pr-2.5 py-1 text-[11px] bg-neutral-950 border border-neutral-700/80 rounded text-neutral-300 placeholder-neutral-500 focus:outline-none focus:border-cyan-500"
            />
          </div>

          <div className="flex items-center rounded border border-neutral-700 bg-neutral-950 p-0.5 text-[10px]">
            {['ALL', 'INFO', 'WARNING', 'SUCCESS', 'ERROR'].map(lvl => (
              <button
                key={lvl}
                onClick={() => setFilterLevel(lvl)}
                className={`px-2 py-0.5 rounded transition-colors ${
                  filterLevel === lvl
                    ? 'bg-neutral-800 text-cyan-300 font-medium'
                    : 'text-neutral-400 hover:text-neutral-200'
                }`}
              >
                {lvl}
              </button>
            ))}
          </div>

          <button
            onClick={() => setAutoScroll(!autoScroll)}
            title="Toggle Auto-Scroll"
            className={`p-1 rounded border transition-colors ${
              autoScroll
                ? 'bg-cyan-500/10 border-cyan-500/40 text-cyan-300'
                : 'border-neutral-700 text-neutral-400 hover:text-neutral-200'
            }`}
          >
            <ArrowDownCircle className="w-3.5 h-3.5" />
          </button>

          <button
            onClick={handleCopy}
            title="Copy Filtered Logs"
            className="p-1 rounded border border-neutral-700 text-neutral-400 hover:text-neutral-200 hover:border-neutral-600 transition-colors"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
          </button>

          {onClear && (
            <button
              onClick={onClear}
              title="Clear View"
              className="p-1 rounded border border-neutral-700 text-neutral-400 hover:text-rose-400 hover:border-rose-800 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      <div
        ref={logContainerRef}
        className={`p-4 font-mono text-[12px] leading-relaxed overflow-y-auto space-y-1.5 scrollbar-thin scrollbar-thumb-neutral-800 ${maxHeight} bg-black/60`}
      >
        {filteredLogs.length === 0 ? (
          <div className="text-neutral-500 italic py-6 text-center">No logs matching query criteria.</div>
        ) : (
          filteredLogs.map(log => (
            <div key={log.id} className="flex items-start gap-2.5 hover:bg-neutral-900/40 px-1.5 py-0.5 rounded transition-colors">
              <span className="text-neutral-500 shrink-0 select-none">[{log.timestamp}]</span>
              <span className={`shrink-0 w-16 uppercase ${getLevelColor(log.level)}`}>[{log.level}]</span>
              <span className="text-neutral-400 shrink-0 max-w-[140px] truncate text-[11px] px-1.5 py-0.2 rounded bg-neutral-900 border border-neutral-800">
                {log.stage}
              </span>
              <span className="text-neutral-200 break-words flex-1">{log.message}</span>
            </div>
          ))
        )}
      </div>

      <div className="px-4 py-1.5 bg-neutral-950 border-t border-neutral-800/80 flex items-center justify-between text-[11px] font-mono text-neutral-500">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span>Stream: Active Local Hook / Mock Telemetry</span>
        </div>
        <div>UTF-8 Log Channel • No Model Weights Exposed</div>
      </div>
    </div>
  );
};
