import { useState, useEffect, useRef, useCallback } from 'react';
import { Pause, Play, Trash2 } from 'lucide-react';
import { useWebSocket } from '../hooks/useWebSocket';
import PageShell from '../components/PageShell';

const UNITS = ['hydrinia', 'moltbot', 'moltbot-voice', 'moltbot-control'];

export default function Logs() {
  const [unit, setUnit] = useState('hydrinia');
  const [lines, setLines] = useState<string[]>([]);
  const [paused, setPaused] = useState(false);
  const [filter, setFilter] = useState('');
  const bottomRef = useRef<HTMLDivElement>(null);

  const onMessage = useCallback((data: { line: string }) => {
    if (!paused && data?.line) {
      setLines(prev => {
        const next = [...prev, data.line];
        return next.length > 2000 ? next.slice(-1500) : next;
      });
    }
  }, [paused]);

  const { connected } = useWebSocket<{ line: string }>(
    `/ws/logs?unit=${unit}&lines=200`,
    onMessage
  );

  useEffect(() => {
    if (!paused) bottomRef.current?.scrollIntoView({ behavior: 'auto' });
  }, [lines, paused]);

  const filtered = filter
    ? lines.filter(l => l.toLowerCase().includes(filter.toLowerCase()))
    : lines;

  const getLineColor = (line: string) => {
    if (line.includes('ERROR') || line.includes('error')) return 'text-red';
    if (line.includes('WARNING') || line.includes('warning')) return 'text-yellow';
    if (line.includes('DEBUG') || line.includes('debug')) return 'text-silver-dim/60';
    return 'text-silver';
  };

  return (
    <PageShell className="logs-page h-full flex flex-col">
      <div className="page-heading-row mb-5">
        <div className="page-heading">
          <p className="page-kicker">Logs</p>
          <h1 className="page-title">Live Event Stream</h1>
        </div>
        <div className="flex items-center gap-3">
          <span className={`status-dot ${connected ? 'status-dot-green' : 'status-dot-red'}`} />
          <span className="text-xs text-silver-dim">{connected ? 'Connected' : 'Disconnected'}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="log-controls">
        <div className="view-tabs" aria-label="Log unit">
          {UNITS.map(u => (
            <button
              key={u}
              type="button"
              aria-pressed={unit === u}
              onClick={() => { setLines([]); setUnit(u); }}
            >
              {u}
            </button>
          ))}
        </div>
        <input
          value={filter}
          onChange={e => setFilter(e.target.value)}
          placeholder="Filter..."
          className="form-control log-filter"
        />
        <button
          type="button"
          onClick={() => setPaused(previous => !previous)}
          className="secondary-button"
        >
          {paused ? <Play size={15} aria-hidden="true" /> : <Pause size={15} aria-hidden="true" />}
          {paused ? 'Resume' : 'Pause'}
        </button>
        <button
          type="button"
          onClick={() => setLines([])}
          className="secondary-button"
        >
          <Trash2 size={15} aria-hidden="true" /> Clear
        </button>
        <span className="text-xs text-silver-dim/60">{filtered.length} lines</span>
      </div>

      {/* Log output */}
      <div className="logs-output">
        <div className="p-3">
          {filtered.map((line, i) => (
            <div key={i} className={`log-line ${getLineColor(line)} hover:bg-bg-input/50 px-1`}>
              {line}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>
    </PageShell>
  );
}
