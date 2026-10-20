import { useState } from 'react';
import { Layers3, OctagonX, Play, RotateCcw, Send, X } from 'lucide-react';
import { usePolling } from '../hooks/usePolling';
import Card from '../components/Card';
import PageShell from '../components/PageShell';
import StatusDot from '../components/StatusDot';
import { EmptyState } from '../components/DataState';
import { post } from '../lib/api';
import { fmtDate } from '../lib/format';

interface HealthResponse {
  services: { name: string; status: string }[];
}

interface Directive {
  type: string;
  expires_at?: string;
  command: string;
  symbol?: string;
}

interface DirectivesResponse {
  directives: Directive[];
}

interface ActionStatus {
  emergency_stopped?: boolean;
}

interface ActionResponse {
  message?: string;
  status?: string;
}

export default function Control() {
  const { data: health, refetch: refetchHealth } = usePolling<HealthResponse>('/api/health/detailed', 15000);
  const { data: directives } = usePolling<DirectivesResponse>('/api/directives', 30000);
  const { data: actionStatus } = usePolling<ActionStatus>('/api/actions/status', 10000);
  const [actionResult, setActionResult] = useState<{ ok: boolean; msg: string } | null>(null);
  const [confirming, setConfirming] = useState<string | null>(null);
  const [directiveForm, setDirectiveForm] = useState({ command: '', type: 'CONSIDER' });

  const doAction = async (action: string, body?: unknown) => {
    setActionResult(null);
    try {
      const res = await post<ActionResponse>(`/api/actions/${action}`, body);
      setActionResult({ ok: true, msg: res.message || res.status || 'Done' });
      setConfirming(null);
      refetchHealth();
    } catch (err) {
      setActionResult({ ok: false, msg: err instanceof Error ? err.message : 'Failed' });
    }
  };

  const restartService = async (svc: string) => {
    try {
      await post(`/api/actions/restart/${svc}`);
      setActionResult({ ok: true, msg: `${svc} restarted` });
      refetchHealth();
    } catch (err) {
      setActionResult({ ok: false, msg: err instanceof Error ? err.message : 'Failed' });
    }
  };

  const issueDirective = async () => {
    if (!directiveForm.command.trim()) return;
    try {
      await post('/api/directives', { command: directiveForm.command, type: directiveForm.type });
      setDirectiveForm({ command: '', type: 'CONSIDER' });
      setActionResult({ ok: true, msg: 'Directive issued' });
    } catch (err) {
      setActionResult({ ok: false, msg: err instanceof Error ? err.message : 'Failed' });
    }
  };

  return (
    <PageShell className="space-y-6">
      <div className="page-heading">
        <p className="page-kicker">Control</p>
        <h1 className="page-title">Command &amp; Control</h1>
        <p className="page-subtitle">Operator-gated service and directive controls.</p>
      </div>

      {actionResult && (
        <div className={`inline-result ${actionResult.ok ? 'text-buy' : 'text-sell'}`} role="status">
          {actionResult.msg}
        </div>
      )}

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Service Control */}
        <Card title="Services">
          <div className="space-y-3">
            {health?.services?.map(svc => (
              <div key={svc.name} className="flex items-center justify-between gap-3 py-3 border-b border-border last:border-b-0">
                <div className="flex items-center gap-3">
                  <StatusDot status={svc.status} />
                  <span className="text-sm font-medium text-silver-light">{svc.name}</span>
                  <span className="text-xs text-silver-dim">{svc.status}</span>
                </div>
                <button
                  onClick={() => restartService(svc.name)}
                  className="secondary-button"
                >
                  <RotateCcw size={14} aria-hidden="true" /> Restart
                </button>
              </div>
            ))}
          </div>
        </Card>

        {/* Quick Actions */}
        <Card title="Quick Actions">
          <div className="space-y-3">
            {/* Emergency Stop */}
            {actionStatus?.emergency_stopped ? (
              <button
                onClick={() => doAction('resume')}
                className="primary-button w-full text-buy"
              >
                <Play size={15} aria-hidden="true" /> Resume Trading
              </button>
            ) : (
              <button
                onClick={() => confirming === 'halt' ? doAction('halt') : setConfirming('halt')}
                className="danger-button w-full"
              >
                <OctagonX size={15} aria-hidden="true" /> {confirming === 'halt' ? 'Confirm Emergency Stop' : 'Emergency Stop'}
              </button>
            )}

            {/* Flatten */}
            <button
              onClick={() => confirming === 'flatten' ? doAction('flatten', { confirmation: 'FLATTEN' }) : setConfirming('flatten')}
              className="secondary-button w-full text-hold"
            >
              <Layers3 size={15} aria-hidden="true" /> {confirming === 'flatten' ? 'Confirm Flatten All Positions' : 'Flatten All Positions'}
            </button>

            {confirming && (
              <button onClick={() => setConfirming(null)} className="secondary-button w-full">
                <X size={14} aria-hidden="true" /> Cancel
              </button>
            )}
          </div>
        </Card>

        {/* Issue Directive */}
        <Card title="Issue Directive">
          <div className="space-y-3">
            <select
              value={directiveForm.type}
              onChange={e => setDirectiveForm(f => ({ ...f, type: e.target.value }))}
              className="form-control"
            >
              <option value="CONSIDER">CONSIDER (48h, informational)</option>
              <option value="VOTE_INFLUENCE">VOTE_INFLUENCE (24h, strong guidance)</option>
              <option value="EXECUTE">EXECUTE (1h, mandatory)</option>
            </select>
            <textarea
              value={directiveForm.command}
              onChange={e => setDirectiveForm(f => ({ ...f, command: e.target.value }))}
              placeholder="Enter directive command..."
              rows={3}
              className="form-control resize-none"
            />
            <button
              onClick={issueDirective}
              disabled={!directiveForm.command.trim()}
              className="primary-button w-full"
            >
              <Send size={15} aria-hidden="true" /> Issue Directive
            </button>
          </div>
        </Card>

        {/* Active Directives */}
        <Card title={`Active Directives (${directives?.directives?.length || 0})`}>
          {directives?.directives?.length ? (
            <div className="space-y-2">
              {directives.directives.map((d, i) => (
                <div key={`${d.type}-${d.expires_at ?? i}`} className="py-3 border-b border-border last:border-b-0 text-sm">
                  <div className="flex justify-between items-center mb-1">
                    <span className={`quiet-meta ${
                      d.type === 'EXECUTE' ? 'text-sell' :
                      d.type === 'VOTE_INFLUENCE' ? 'text-hold' :
                      'text-accent'
                    }`}>{d.type}</span>
                    <span className="text-xs text-silver-dim">Expires: {fmtDate(d.expires_at)}</span>
                  </div>
                  <p className="text-silver text-sm mt-1">{d.command}</p>
                  {d.symbol && <p className="text-xs text-silver-dim mt-1">Symbol: {d.symbol}</p>}
                </div>
              ))}
            </div>
          ) : <EmptyState>No active directives</EmptyState>}
        </Card>
      </div>
    </PageShell>
  );
}
