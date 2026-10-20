import { useMemo } from 'react';
import { usePolling } from '../hooks/usePolling';
import Card from '../components/Card';
import PageShell from '../components/PageShell';
import Signal from '../components/Signal';
import { EmptyState, Freshness, PanelSkeleton, RetryNote } from '../components/DataState';
import { fmtPct, fmtMoney, fmtDate, plColor } from '../lib/format';

interface SourcedDatum {
  value?: string | number | null;
  as_of?: string;
  updated?: string;
  source?: string;
  stale?: boolean;
}

interface RegimeResponse {
  regime?: string;
  posture?: string;
  signal?: string;
  confidence?: number;
  vix?: SourcedDatum | number | null;
  vix_value?: number | null;
  vix_as_of?: string;
  vix_source?: string;
  vix_stale?: boolean;
  sentiment?: SourcedDatum | string | number | null;
  sentiment_value?: string | number | null;
  sentiment_as_of?: string;
  sentiment_source?: string;
  sentiment_stale?: boolean;
  as_of?: string;
  updated?: string;
  stale?: boolean;
}

interface RiskResponse {
  daily_metrics?: {
    daily_pnl?: number;
    daily_return_pct?: number;
    trades_today?: number;
    win_rate?: number;
  };
  max_concentration_pct?: number;
  as_of?: string;
  stale?: boolean;
}

interface Thesis {
  symbol?: string;
  ticker?: string;
  category?: string;
  direction?: string;
  signal?: string;
  thesis?: string;
  rationale?: string;
  conviction?: number | string;
}

interface AgentStats {
  model?: string;
  requested_model?: string;
  served_model?: string;
  served_model_verified?: boolean;
  platform?: string;
  model_source?: string;
  source?: string;
  source_age?: string | number;
  as_of?: string;
  updated?: string;
  last_updated?: string;
  signal?: string;
  recommendation?: string;
  vote?: string;
  total_votes?: number;
  buy_votes?: number;
  sell_votes?: number;
  hold_votes?: number;
  accuracy?: number;
  vote_skew?: number;
}

interface AlertItem {
  id?: string | number;
  severity?: string;
  type?: string;
  category?: string;
  message?: string;
  timestamp?: string;
  created_at?: string;
  updated?: string;
  status?: string;
  active?: boolean;
}

interface AlertsResponse {
  alerts?: AlertItem[];
  active?: AlertItem[];
  active_alerts?: AlertItem[];
  diagnostics?: AlertItem[];
  diagnostic_alerts?: AlertItem[];
  as_of?: string;
  stale?: boolean;
}

interface Trade {
  id?: string;
  filled_at?: string;
  submitted_at?: string;
  symbol: string;
  side?: string;
  filled_qty?: number | string;
  qty?: number | string;
  filled_avg_price?: number | string;
  status?: string;
}

function sourcedDatum(
  raw: SourcedDatum | string | number | null | undefined,
  fallback: { value?: string | number | null; asOf?: string; source?: string; stale?: boolean },
) {
  if (raw && typeof raw === 'object') {
    return {
      value: raw.value ?? null,
      asOf: raw.as_of ?? raw.updated,
      source: raw.source ?? fallback.source,
      stale: raw.stale ?? fallback.stale ?? false,
    };
  }
  return { value: raw ?? fallback.value ?? null, asOf: fallback.asOf, source: fallback.source, stale: fallback.stale ?? false };
}

function sourceAge(value?: string | number): string {
  if (typeof value === 'string' && !Number.isFinite(Date.parse(value))) return value;
  const date = value == null ? null : new Date(value);
  if (!date || Number.isNaN(date.getTime())) return 'age unavailable';
  const seconds = Math.max(0, (Date.now() - date.getTime()) / 1000);
  if (seconds < 60) return 'under 1m old';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m old`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h old`;
  return `${Math.floor(seconds / 86400)}d old`;
}

function alertSeverity(alert: AlertItem): 'critical' | 'warning' | 'info' {
  const severity = String(alert.severity ?? '').toLowerCase();
  if (['critical', 'error', 'emergency', 'high'].includes(severity)) return 'critical';
  if (['warning', 'warn', 'medium'].includes(severity)) return 'warning';
  return 'info';
}

function AlertRows({ items }: { items: AlertItem[] }) {
  return (
    <div className="alert-list">
      {[...items].reverse().map((alert, index) => {
        const severity = alertSeverity(alert);
        const timestamp = alert.timestamp ?? alert.created_at ?? alert.updated;
        const message = alert.message ?? 'No diagnostic message supplied';
        return (
          <div key={alert.id ?? `${timestamp ?? 'alert'}-${index}`} className={`alert-row alert-${severity}`}>
            <span className="alert-level">{alert.type ?? alert.category ?? severity}</span>
            <span className="alert-message" title={message}>{message}</span>
            <time className="alert-time" dateTime={timestamp}>{timestamp ? fmtDate(timestamp) : '--'}</time>
          </div>
        );
      })}
    </div>
  );
}

export default function Hydrinia() {
  const riskQuery = usePolling<RiskResponse>('/api/hydrinia/risk', 30000);
  const regimeQuery = usePolling<RegimeResponse>('/api/hydrinia/regime', 60000);
  const thesesQuery = usePolling<{ theses: Thesis[]; as_of?: string; stale?: boolean }>('/api/hydrinia/theses', 60000);
  const agentsQuery = usePolling<{ agents: Record<string, AgentStats>; as_of?: string; stale?: boolean }>('/api/hydrinia/agents', 60000);
  const alertsQuery = usePolling<AlertsResponse>('/api/hydrinia/alerts', 15000);
  const tradesQuery = usePolling<{ trades: Trade[]; as_of?: string; stale?: boolean }>('/api/trades?limit=10', 30000);

  const risk = riskQuery.data;
  const regime = regimeQuery.data;
  const alerts = alertsQuery.data;
  const winRate = risk?.daily_metrics?.win_rate ?? null;
  const winRateDisplay = winRate == null ? null : winRate > 1 ? winRate : winRate * 100;

  const vix = sourcedDatum(regime?.vix, { value: regime?.vix_value, asOf: regime?.vix_as_of, source: regime?.vix_source, stale: regime?.vix_stale });
  const sentiment = sourcedDatum(regime?.sentiment, { value: regime?.sentiment_value, asOf: regime?.sentiment_as_of, source: regime?.sentiment_source, stale: regime?.sentiment_stale });

  const { activeAlerts, diagnostics } = useMemo(() => {
    const all = alerts?.alerts ?? [];
    const diagnosticFallback = all.filter(alert => {
      const descriptor = `${alert.type ?? ''} ${alert.category ?? ''} ${alert.status ?? ''}`.toLowerCase();
      return alert.active === false || descriptor.includes('diagnostic') || descriptor.includes('resolved') || descriptor.includes('inactive');
    });
    const explicitDiagnostics = alerts?.diagnostics ?? alerts?.diagnostic_alerts;
    const diagnosticItems = explicitDiagnostics ?? diagnosticFallback;
    const explicitActive = alerts?.active_alerts ?? alerts?.active;
    const activeItems = explicitActive ?? all.filter(alert => !diagnosticItems.includes(alert));
    return { activeAlerts: activeItems, diagnostics: diagnosticItems };
  }, [alerts]);

  const agents = Object.entries(agentsQuery.data?.agents ?? {}).sort(([, left], [, right]) => (right.total_votes ?? 0) - (left.total_votes ?? 0));

  return (
    <PageShell className="space-y-6">
      <div className="page-heading-row">
        <div className="page-heading">
          <p className="page-kicker">Hydrinia</p>
          <h1 className="page-title">Autonomy Intelligence</h1>
          <p className="page-subtitle">Cached market context, active risk signals, agent provenance, and execution evidence.</p>
        </div>
        <Freshness stale={regime?.stale || regimeQuery.isStale} asOf={regime?.as_of ?? regime?.updated} source="Regime cache" />
      </div>

      <div className="section-grid-two">
        <Card title="Market Regime">
          {regimeQuery.loading && !regime ? <PanelSkeleton /> : regime ? (
            <dl className="key-value-grid">
              <div className="key-value-row"><dt>Regime</dt><dd><span className="mr-2">{regime.regime ?? 'Unavailable'}</span><Signal value={regime.posture ?? regime.signal ?? regime.regime} /></dd></div>
              <div className="key-value-row"><dt>Confidence</dt><dd>{regime.confidence == null ? 'Unavailable' : fmtPct(regime.confidence <= 1 ? regime.confidence * 100 : regime.confidence, 0)}</dd></div>
              <div className="key-value-row">
                <dt>VIX</dt>
                <dd>{vix.value == null ? 'Unavailable' : vix.value}<span className="source-age">{vix.source ?? 'Source unavailable'} / {sourceAge(vix.asOf)}</span></dd>
              </div>
              <div className="key-value-row">
                <dt>Sentiment</dt>
                <dd>{sentiment.value == null ? 'Unavailable' : sentiment.value}<span className="source-age">{sentiment.source ?? 'Source unavailable'} / {sourceAge(sentiment.asOf)}</span></dd>
              </div>
            </dl>
          ) : <EmptyState>Regime cache unavailable</EmptyState>}
          <div className="flex flex-wrap gap-2 mt-3">
            {vix.value != null ? <Freshness stale={vix.stale} asOf={vix.asOf} source={vix.source ?? 'VIX'} /> : null}
            {sentiment.value != null ? <Freshness stale={sentiment.stale} asOf={sentiment.asOf} source={sentiment.source ?? 'Sentiment'} /> : null}
          </div>
          <RetryNote error={regimeQuery.error} hasData={Boolean(regime)} />
        </Card>

        <Card title="Risk Metrics" action={risk ? <Freshness stale={risk.stale || riskQuery.isStale} asOf={risk.as_of} /> : undefined}>
          {riskQuery.loading && !risk ? <PanelSkeleton /> : risk?.daily_metrics ? (
            <dl className="key-value-grid">
              <div className="key-value-row"><dt>Daily P&amp;L</dt><dd className={plColor(risk.daily_metrics.daily_pnl)}>{fmtMoney(risk.daily_metrics.daily_pnl)}</dd></div>
              <div className="key-value-row"><dt>Daily Return</dt><dd className={plColor(risk.daily_metrics.daily_return_pct)}>{fmtPct(risk.daily_metrics.daily_return_pct)}</dd></div>
              <div className="key-value-row"><dt>Trades Today</dt><dd>{risk.daily_metrics.trades_today ?? '--'}</dd></div>
              <div className="key-value-row"><dt>Win Rate</dt><dd>{winRateDisplay == null ? '--' : fmtPct(winRateDisplay, 0)}</dd></div>
              <div className="key-value-row"><dt>Max Concentration</dt><dd>{fmtPct(risk.max_concentration_pct, 1)}</dd></div>
            </dl>
          ) : <EmptyState>Risk metrics unavailable</EmptyState>}
          <RetryNote error={riskQuery.error} hasData={Boolean(risk)} />
        </Card>
      </div>

      <div className="hydrinia-alert-grid">
        <Card title={`Active Alerts (${activeAlerts.length})`} action={alerts ? <Freshness stale={alerts.stale || alertsQuery.isStale} asOf={alerts.as_of} /> : undefined}>
          {alertsQuery.loading && !alerts ? <PanelSkeleton /> : activeAlerts.length ? <AlertRows items={activeAlerts} /> : <EmptyState>No active alerts</EmptyState>}
          <RetryNote error={alertsQuery.error} hasData={activeAlerts.length > 0} />
        </Card>
        <Card title={`Diagnostics (${diagnostics.length})`} className="diagnostics-panel">
          {alertsQuery.loading && !alerts ? <PanelSkeleton /> : diagnostics.length ? <AlertRows items={diagnostics} /> : <EmptyState>No diagnostics reported</EmptyState>}
        </Card>
      </div>

      <Card title="Active Theses" action={thesesQuery.data ? <Freshness stale={thesesQuery.data.stale || thesesQuery.isStale} asOf={thesesQuery.data.as_of} /> : undefined}>
        {thesesQuery.loading && !thesesQuery.data ? <PanelSkeleton /> : thesesQuery.data?.theses?.length ? (
          <div className="thesis-list">
            {thesesQuery.data.theses.map((thesis, index) => (
              <article key={`${thesis.symbol ?? thesis.ticker ?? 'thesis'}-${index}`} className="thesis-row">
                <div className="thesis-head"><strong className="text-silver-light">{thesis.symbol ?? thesis.ticker ?? 'Unknown'}</strong><Signal value={thesis.signal ?? thesis.direction} /></div>
                {thesis.category ? <p className="quiet-meta mt-2">{thesis.category}</p> : null}
                <p className="thesis-copy">{thesis.thesis ?? thesis.rationale ?? 'No thesis detail supplied'}</p>
                {thesis.conviction != null ? <p className="source-age">Conviction: {typeof thesis.conviction === 'number' ? fmtPct(thesis.conviction <= 1 ? thesis.conviction * 100 : thesis.conviction, 0) : thesis.conviction}</p> : null}
              </article>
            ))}
          </div>
        ) : <EmptyState>No active theses</EmptyState>}
        <RetryNote error={thesesQuery.error} hasData={Boolean(thesesQuery.data?.theses?.length)} />
      </Card>

      <div className="section-grid-two">
        <Card title="Agent Performance" action={agentsQuery.data ? <Freshness stale={agentsQuery.data.stale || agentsQuery.isStale} asOf={agentsQuery.data.as_of} /> : undefined}>
          {agentsQuery.loading && !agentsQuery.data ? <PanelSkeleton /> : agents.length ? (
            <div className="table-scroll">
              <table className="mobile-table">
                <thead><tr><th className="py-2 px-2 text-left">Agent</th><th className="py-2 px-2 text-left">Model / Source</th><th className="py-2 px-2 text-left">Signal</th><th className="py-2 px-2 text-right">Votes</th><th className="py-2 px-2 text-right">Buy</th><th className="py-2 px-2 text-right">Sell</th><th className="py-2 px-2 text-right">Hold</th><th className="py-2 px-2 text-right">Skew</th><th className="py-2 px-2 text-right">Accuracy</th></tr></thead>
                <tbody>
                  {agents.map(([name, agent]) => {
                    const agentTime = agent.source_age ?? agent.as_of ?? agent.updated ?? agent.last_updated;
                    const model = agent.served_model ?? agent.requested_model ?? agent.model ?? 'Unspecified';
                    const modelEvidence = agent.served_model_verified ? 'served model' : 'requested model';
                    return (
                      <tr key={name}>
                        <td className="py-2 px-2" data-primary="true"><span className="instrument-label">{name}</span></td>
                        <td className="py-2 px-2" data-label="Model / Source"><span>{model}</span><span className="source-age">{agent.platform ?? agent.source ?? 'Source unavailable'} / {modelEvidence} / {sourceAge(agentTime)}</span></td>
                        <td className="py-2 px-2" data-label="Signal"><Signal value={agent.signal ?? agent.recommendation ?? agent.vote} /></td>
                        <td className="py-2 px-2 text-right" data-label="Votes">{agent.total_votes ?? 0}</td>
                        <td className="py-2 px-2 text-right text-buy" data-label="Buy">{agent.buy_votes ?? 0}</td>
                        <td className="py-2 px-2 text-right text-sell" data-label="Sell">{agent.sell_votes ?? 0}</td>
                        <td className="py-2 px-2 text-right text-hold" data-label="Hold">{agent.hold_votes ?? 0}</td>
                        <td className="py-2 px-2 text-right" data-label="Skew"><span>{agent.vote_skew == null ? '--' : fmtPct(agent.vote_skew * 100, 0)}</span></td>
                        <td className="py-2 px-2 text-right" data-label="Accuracy">{agent.accuracy == null ? '--' : fmtPct(agent.accuracy <= 1 ? agent.accuracy * 100 : agent.accuracy, 1)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : <EmptyState>No agent data available</EmptyState>}
          <RetryNote error={agentsQuery.error} hasData={agents.length > 0} />
        </Card>

        <Card title="Recent Trades" action={tradesQuery.data ? <Freshness stale={tradesQuery.data.stale || tradesQuery.isStale} asOf={tradesQuery.data.as_of} /> : undefined}>
          {tradesQuery.loading && !tradesQuery.data ? <PanelSkeleton /> : tradesQuery.data?.trades?.length ? (
            <div className="table-scroll"><table className="mobile-table"><thead><tr><th className="py-2 px-2 text-left">Time</th><th className="py-2 px-2 text-left">Symbol</th><th className="py-2 px-2 text-left">Side</th><th className="py-2 px-2 text-right">Qty</th><th className="py-2 px-2 text-right">Price</th><th className="py-2 px-2 text-left">Status</th></tr></thead><tbody>{tradesQuery.data.trades.map((trade, index) => <tr key={trade.id ?? index}><td className="py-2 px-2" data-label="Time">{fmtDate(trade.filled_at ?? trade.submitted_at)}</td><td className="py-2 px-2" data-primary="true"><span className="instrument-label">{trade.symbol}</span></td><td className="py-2 px-2" data-label="Side"><Signal value={trade.side} /></td><td className="py-2 px-2 text-right" data-label="Qty">{trade.filled_qty ?? trade.qty ?? '--'}</td><td className="py-2 px-2 text-right" data-label="Price">{fmtMoney(Number(trade.filled_avg_price ?? 0))}</td><td className="py-2 px-2" data-label="Status">{trade.status ?? '--'}</td></tr>)}</tbody></table></div>
          ) : <EmptyState>No recent trades</EmptyState>}
          <RetryNote error={tradesQuery.error} hasData={Boolean(tradesQuery.data?.trades?.length)} />
        </Card>
      </div>
    </PageShell>
  );
}
