import { useMemo } from 'react';
import { LockKeyhole } from 'lucide-react';
import { usePolling } from '../hooks/usePolling';
import Card, { StatCard } from '../components/Card';
import StatusDot from '../components/StatusDot';
import PageShell from '../components/PageShell';
import Signal from '../components/Signal';
import { Freshness, PanelSkeleton, RetryNote, EmptyState } from '../components/DataState';
import { fmtMoney, fmtPct, plColor, fmtNum } from '../lib/format';

interface Position {
  symbol: string;
  qty: number;
  side: string;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  asset_class?: string;
}

interface Portfolio {
  account: {
    equity: number;
    cash: number;
    buying_power: number;
    day_pl: number;
    day_pl_pct: number;
    status: string;
    pattern_day_trader: boolean;
    trading_blocked: boolean;
  };
  positions: Position[];
  as_of?: string;
  stale?: boolean;
}

interface Health {
  services: { name: string; status: string }[];
  system: {
    cpu_percent: number;
    memory_percent: number;
    memory_used_gb: number;
    memory_total_gb: number;
    disk_percent: number;
  };
  as_of?: string;
  stale?: boolean;
}

interface Risk {
  equity: number;
  exposure: { long: number; short: number; symbols: Record<string, number> };
  max_concentration_pct: number;
  deployed_pct?: number;
  drawdown_tier?: string;
  current_drawdown_tier?: string;
  variance_budget_pct?: number;
  variance_budget?: { remaining_pct?: number };
  daily_metrics: {
    daily_pnl: number;
    daily_return_pct: number;
    trades_today: number;
    win_rate: number;
  };
  as_of?: string;
  stale?: boolean;
}

interface Regime {
  regime?: string;
  posture?: string;
  signal?: string;
  as_of?: string;
  updated?: string;
  stale?: boolean;
}

function formatOccLabel(symbol: string): string {
  const match = symbol.match(/^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/);
  if (!match) return symbol;
  const [, root, , month, day, type, strikeRaw] = match;
  const strike = parseInt(strikeRaw, 10) / 1000;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${root} ${months[parseInt(month, 10) - 1]} ${day} $${strike % 1 === 0 ? strike.toFixed(0) : strike.toFixed(2)} ${type === 'C' ? 'Call' : 'Put'}`;
}

export default function Dashboard() {
  const portfolioQuery = usePolling<Portfolio>('/api/portfolio', 15000);
  const healthQuery = usePolling<Health>('/api/health/detailed', 30000);
  const riskQuery = usePolling<Risk>('/api/hydrinia/risk', 30000);
  const regimeQuery = usePolling<Regime>('/api/hydrinia/regime', 60000);

  const portfolio = portfolioQuery.data;
  const health = healthQuery.data;
  const risk = riskQuery.data;
  const regime = regimeQuery.data;
  const account = portfolio?.account;

  const { equities, options } = useMemo(() => {
    const equityPositions: Position[] = [];
    const optionPositions: Position[] = [];
    for (const position of portfolio?.positions ?? []) {
      if (String(position.asset_class ?? '').toLowerCase().includes('option')) optionPositions.push(position);
      else equityPositions.push(position);
    }
    return { equities: equityPositions, options: optionPositions };
  }, [portfolio?.positions]);

  const deployedPct = useMemo(() => {
    if (risk?.deployed_pct != null) return risk.deployed_pct;
    const equity = account?.equity ?? risk?.equity;
    if (!equity || !risk?.exposure) return null;
    return ((Math.abs(risk.exposure.long ?? 0) + Math.abs(risk.exposure.short ?? 0)) / equity) * 100;
  }, [account?.equity, risk]);

  const varianceBudget = risk?.variance_budget_pct ?? risk?.variance_budget?.remaining_pct;
  const drawdownTier = risk?.drawdown_tier ?? risk?.current_drawdown_tier ?? 'Unavailable';
  const regimeName = regime?.regime ?? 'Unavailable';
  const regimeSignal = regime?.posture ?? regime?.signal ?? regime?.regime;

  const positionRows = (positions: Position[], isOption: boolean) => positions.map(position => (
    <tr key={position.symbol}>
      <td className="py-3 px-3" data-primary="true">
        <span className="instrument-label" title={formatOccLabel(position.symbol)}>
          {isOption ? formatOccLabel(position.symbol) : position.symbol}
        </span>
        {isOption ? <span className="instrument-code" title={position.symbol}>{position.symbol}</span> : null}
      </td>
      <td className="py-3 px-3 text-right" data-label="Qty">{fmtNum(position.qty)}</td>
      <td className="py-3 px-3 text-right" data-label="Entry">{fmtMoney(position.avg_entry_price)}</td>
      <td className="py-3 px-3 text-right" data-label="Current">{fmtMoney(position.current_price)}</td>
      <td className="py-3 px-3 text-right" data-label="Market Value">{fmtMoney(position.market_value)}</td>
      <td className={`py-3 px-3 text-right ${plColor(position.unrealized_pl)}`} data-label="Unrealized P&L">{fmtMoney(position.unrealized_pl)}</td>
      <td className={`py-3 px-3 text-right ${plColor(position.unrealized_plpc)}`} data-label="Return">{fmtPct(position.unrealized_plpc)}</td>
    </tr>
  ));

  const positionHeader = (
    <tr>
      <th className="py-3 px-3 text-left">Instrument</th>
      <th className="py-3 px-3 text-right">Qty</th>
      <th className="py-3 px-3 text-right">Entry</th>
      <th className="py-3 px-3 text-right">Current</th>
      <th className="py-3 px-3 text-right">Market Value</th>
      <th className="py-3 px-3 text-right">Unrealized P&amp;L</th>
      <th className="py-3 px-3 text-right">Return</th>
    </tr>
  );

  return (
    <PageShell className="space-y-6">
      <div className="page-heading-row">
        <div className="page-heading">
          <p className="page-kicker">Overview</p>
          <h1 className="page-title">Mission Control</h1>
          <p className="page-subtitle">Capital posture, risk, service health, and open exposure from the guarded soak environment.</p>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="quiet-meta inline-flex items-center gap-2"><LockKeyhole size={14} aria-hidden="true" /> Execution gate <strong className="status-warning">off</strong></span>
          <Freshness stale={portfolio?.stale || portfolioQuery.isStale} asOf={portfolio?.as_of} source="Portfolio" />
        </div>
      </div>

      {portfolioQuery.loading && !portfolio ? <PanelSkeleton /> : (
        <div className="overview-kpi-grid" aria-label="Portfolio and risk key performance indicators">
          <StatCard label="Equity" value={fmtMoney(account?.equity)} />
          <StatCard label="Day P&L" value={fmtMoney(account?.day_pl)} sub={fmtPct(account?.day_pl_pct)} color={plColor(account?.day_pl)} />
          <StatCard label="Deployed" value={deployedPct == null ? 'Unavailable' : `${deployedPct.toFixed(1)}%`} />
          <StatCard label="Drawdown Tier" value={drawdownTier} color={drawdownTier === 'Unavailable' ? 'text-silver-dim' : 'text-hold'} />
          <StatCard label="Variance Budget" value={varianceBudget == null ? 'Unavailable' : `${varianceBudget.toFixed(1)}%`} />
          <article className="institutional-card stat-card">
            <p className="label">Regime</p>
            <p className="metric-value">{regimeName}</p>
            <div className="stat-subline"><Signal value={regimeSignal} /></div>
          </article>
        </div>
      )}
      <RetryNote error={portfolioQuery.error || riskQuery.error || regimeQuery.error} hasData={Boolean(portfolio || risk || regime)} />

      <div className="section-grid-three">
        <Card title={`Equity Positions (${equities.length})`} className="xl:col-span-2">
          {equities.length ? (
            <div className="table-scroll">
              <table className="mobile-table">
                <thead>{positionHeader}</thead>
                <tbody>{positionRows(equities, false)}</tbody>
              </table>
            </div>
          ) : <EmptyState>No equity positions</EmptyState>}
        </Card>

        <div className="space-y-4">
          <Card title="System Health" action={<Freshness stale={health?.stale || healthQuery.isStale} asOf={health?.as_of} />}>
            {healthQuery.loading && !health ? <PanelSkeleton /> : health?.system ? (
              <dl className="key-value-grid">
                <div className="key-value-row"><dt>CPU</dt><dd>{health.system.cpu_percent}%</dd></div>
                <div className="key-value-row"><dt>Memory</dt><dd>{health.system.memory_used_gb?.toFixed(1)} / {health.system.memory_total_gb?.toFixed(1)} GB</dd></div>
                <div className="key-value-row"><dt>Disk</dt><dd>{health.system.disk_percent}%</dd></div>
              </dl>
            ) : <EmptyState>System metrics unavailable</EmptyState>}
            {health?.services?.length ? (
              <div className="mt-3 pt-3 border-t border-border space-y-2">
                {health.services.map(service => (
                  <div key={service.name} className="flex items-center justify-between gap-3 text-sm">
                    <span className="text-silver-dim">{service.name}</span>
                    <span className="inline-flex items-center gap-2 text-silver"><StatusDot status={service.status} /> {service.status}</span>
                  </div>
                ))}
              </div>
            ) : null}
          </Card>

          <Card title="Exposure" action={<Freshness stale={risk?.stale || riskQuery.isStale} asOf={risk?.as_of} />}>
            {riskQuery.loading && !risk ? <PanelSkeleton /> : risk?.exposure ? (
              <dl className="key-value-grid">
                <div className="key-value-row"><dt>Long</dt><dd className="text-buy">{fmtMoney(risk.exposure.long)}</dd></div>
                <div className="key-value-row"><dt>Short</dt><dd className="text-sell">{fmtMoney(risk.exposure.short)}</dd></div>
                <div className="key-value-row"><dt>Max Concentration</dt><dd>{fmtPct(risk.max_concentration_pct, 1)}</dd></div>
              </dl>
            ) : <EmptyState>Exposure data unavailable</EmptyState>}
          </Card>
        </div>
      </div>

      {options.length ? (
        <Card title={`Option Positions (${options.length})`}>
          <div className="table-scroll">
            <table className="mobile-table">
              <thead>{positionHeader}</thead>
              <tbody>{positionRows(options, true)}</tbody>
            </table>
          </div>
        </Card>
      ) : null}
    </PageShell>
  );
}
