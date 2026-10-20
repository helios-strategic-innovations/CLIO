import { Lightbulb } from 'lucide-react';
import { Bar, BarChart, Cell, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import Card, { StatCard } from '../components/Card';
import PageShell from '../components/PageShell';
import { EmptyState, Freshness, PanelSkeleton, RetryNote } from '../components/DataState';
import { usePolling } from '../hooks/usePolling';

interface Metrics {
  sharpe: number | null;
  sortino: number | null;
  calmar: number | null;
  max_dd_pct: number | null;
  max_dd_peak: string | null;
  max_dd_trough: string | null;
  total_return_pct: number | null;
  ann_return_pct: number | null;
  ann_vol_pct: number | null;
  best_day_pct: number | null;
  worst_day_pct: number | null;
  up_days: number;
  down_days: number;
  avg_daily_pct: number | null;
  start_equity: number | null;
  end_equity: number | null;
  days: number;
  stale?: boolean;
  reason_code?: string;
}

interface Benchmark {
  name: string;
  sharpe: number | null;
  sortino: number | null;
  ours?: boolean;
}

interface AnalyticsData {
  equity_curve: { date: string; equity: number }[];
  metrics: Metrics;
  phases: Record<string, Metrics>;
  benchmarks: Benchmark[];
  fun_facts: string[];
  updated: string;
  source?: string;
  stale?: boolean;
  error?: string;
}

const isMetric = (value: number | null | undefined): value is number => typeof value === 'number' && Number.isFinite(value);
const format = (value: number | null | undefined, decimals = 1) => isMetric(value) ? value.toFixed(decimals) : '--';
const moneyAxis = (value: number) => value >= 1000 ? `$${(value / 1000).toFixed(0)}k` : `$${value.toFixed(0)}`;
const signedPercent = (value: number | null | undefined, decimals = 1) => isMetric(value) ? `${value >= 0 ? '+' : ''}${format(value, decimals)}%` : '--';
const equityRange = (start: number | null, end: number | null) => isMetric(start) && isMetric(end) ? `$${start.toLocaleString()} to $${end.toLocaleString()}` : 'Unavailable';

function dateLabel(value: string): string {
  const date = new Date(`${value}T00:00:00`);
  return Number.isNaN(date.getTime()) ? value : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

function metricTone(value: number | null | undefined): string {
  if (!isMetric(value)) return 'text-silver-dim';
  if (value > 0) return 'text-green';
  if (value < 0) return 'text-red';
  return 'text-silver-dim';
}

function ComparisonChart({ data, metric, label }: { data: Benchmark[]; metric: 'sharpe' | 'sortino'; label: string }) {
  if (!data.length) return <EmptyState>No benchmark data available</EmptyState>;
  return (
    <div className="chart-frame analytics-bar-chart" aria-label={`${label} benchmark chart`}>
      <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1, height: 1 }}>
        <BarChart data={data} layout="vertical" margin={{ left: 12, right: 8, top: 4, bottom: 4 }}>
          <XAxis type="number" tick={{ fill: '#9f998f', fontSize: 11 }} axisLine={{ stroke: 'rgba(232,220,186,0.16)' }} tickLine={false} />
          <YAxis type="category" dataKey="name" width={106} tick={{ fill: '#cbc6bc', fontSize: 11 }} axisLine={false} tickLine={false} />
          <Tooltip contentStyle={{ background: '#201f1d', border: '1px solid rgba(232,220,186,0.3)', borderRadius: 4, color: '#eeeae2', fontSize: 13 }} formatter={value => [typeof value === 'number' ? value.toFixed(2) : '--', label]} />
          <Bar dataKey={metric} isAnimationActive={false}>
            {data.map(benchmark => <Cell key={benchmark.name} fill={benchmark.ours ? '#d6b867' : '#817c73'} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

export default function Analytics() {
  const analyticsQuery = usePolling<AnalyticsData>('/api/analytics', 300000);
  const data = analyticsQuery.data;

  if (analyticsQuery.loading && !data) {
    return <PageShell className="space-y-6"><div className="page-heading"><p className="page-kicker">Analytics</p><h1 className="page-title">Performance Analytics</h1></div><PanelSkeleton /></PageShell>;
  }

  if (!data || !data.metrics) {
    return <PageShell className="space-y-6"><div className="page-heading"><p className="page-kicker">Analytics</p><h1 className="page-title">Performance Analytics</h1></div><EmptyState>Performance analytics unavailable</EmptyState><RetryNote error={analyticsQuery.error ?? data?.error ?? null} /></PageShell>;
  }

  const metrics = data.metrics;
  const periodStart = data.equity_curve[0]?.date;
  const periodEnd = data.equity_curve[data.equity_curve.length - 1]?.date;

  return (
    <PageShell className="space-y-6">
      <div className="page-heading-row">
        <div className="page-heading">
          <p className="page-kicker">Analytics</p>
          <h1 className="page-title">Performance Analytics</h1>
          <p className="page-subtitle">{metrics.days} trading days{periodStart && periodEnd ? `, ${dateLabel(periodStart)} to ${dateLabel(periodEnd)}` : ''}.</p>
        </div>
        <Freshness stale={data.stale || analyticsQuery.isStale} asOf={data.updated} source={data.source ?? 'Alpaca history'} />
      </div>

      <div className="overview-kpi-grid">
        <StatCard label="Sharpe Ratio" value={format(metrics.sharpe, 2)} sub={isMetric(metrics.sharpe) ? (metrics.sharpe >= 2 ? 'Above 2.0' : 'Below 2.0') : 'Insufficient history'} color={isMetric(metrics.sharpe) && metrics.sharpe >= 2 ? 'text-green' : 'text-silver-light'} />
        <StatCard label="Sortino Ratio" value={format(metrics.sortino, 2)} sub={isMetric(metrics.sortino) ? 'Downside-adjusted' : 'Insufficient downside history'} color={isMetric(metrics.sortino) && metrics.sortino >= 2 ? 'text-green' : 'text-silver-light'} />
        <StatCard label="Total Return" value={signedPercent(metrics.total_return_pct)} sub={equityRange(metrics.start_equity, metrics.end_equity)} color={metricTone(metrics.total_return_pct)} />
        <StatCard label="Max Drawdown" value={isMetric(metrics.max_dd_pct) ? `-${format(Math.abs(metrics.max_dd_pct))}%` : '--'} sub={`${metrics.max_dd_peak || '?'} to ${metrics.max_dd_trough || '?'}`} color={isMetric(metrics.max_dd_pct) ? 'text-red' : 'text-silver-dim'} />
        <StatCard label="Calmar Ratio" value={isMetric(metrics.calmar) && metrics.calmar > 999 ? '999+' : format(metrics.calmar, 1)} sub="Return / max drawdown" color={isMetric(metrics.calmar) && metrics.calmar >= 3 ? 'text-yellow' : 'text-silver-light'} />
        <StatCard label="Annual Volatility" value={isMetric(metrics.ann_vol_pct) ? `${format(metrics.ann_vol_pct)}%` : '--'} sub={`Best ${signedPercent(metrics.best_day_pct)} / Worst ${signedPercent(metrics.worst_day_pct)}`} />
      </div>

      <Card title="Equity Curve">
        {data.equity_curve.length ? (
          <div className="chart-frame" aria-label="Analytics equity curve">
            <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1, height: 1 }}>
              <LineChart data={data.equity_curve} margin={{ top: 8, right: 8, left: 2, bottom: 0 }}>
                <XAxis dataKey="date" tickFormatter={dateLabel} tick={{ fill: '#9f998f', fontSize: 11 }} axisLine={{ stroke: 'rgba(232,220,186,0.16)' }} tickLine={false} minTickGap={30} />
                <YAxis tickFormatter={moneyAxis} tick={{ fill: '#9f998f', fontSize: 11 }} axisLine={false} tickLine={false} domain={['auto', 'auto']} width={58} />
                <Tooltip contentStyle={{ background: '#201f1d', border: '1px solid rgba(232,220,186,0.3)', borderRadius: 4, color: '#eeeae2', fontSize: 13 }} labelFormatter={value => dateLabel(String(value))} formatter={value => [`$${Number(value).toLocaleString()}`, 'Equity']} />
                <Line type="monotone" dataKey="equity" stroke="#d6b867" strokeWidth={2} dot={false} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : <EmptyState>No equity history available</EmptyState>}
      </Card>

      <div className="section-grid-two">
        <Card title="Sharpe Ratio vs Industry"><ComparisonChart data={data.benchmarks} metric="sharpe" label="Sharpe" /></Card>
        <Card title="Sortino Ratio vs Industry"><ComparisonChart data={data.benchmarks} metric="sortino" label="Sortino" /></Card>
      </div>

      {Object.keys(data.phases).length ? (
        <Card title="Performance by Phase">
          <div className="table-scroll">
            <table className="mobile-table">
              <thead><tr><th className="py-2 px-3 text-left">Phase</th><th className="py-2 px-3 text-right">Days</th><th className="py-2 px-3 text-right">Return</th><th className="py-2 px-3 text-right">Sharpe</th><th className="py-2 px-3 text-right">Sortino</th><th className="py-2 px-3 text-right">Max DD</th><th className="py-2 px-3 text-right">Vol</th><th className="py-2 px-3 text-right">Best Day</th><th className="py-2 px-3 text-right">Worst Day</th><th className="py-2 px-3 text-right">Up / Down</th></tr></thead>
              <tbody>
                {Object.entries(data.phases).map(([label, phase]) => (
                  <tr key={label}>
                    <td className="py-2 px-3" data-primary="true"><span className="instrument-label">{label}</span></td><td className="py-2 px-3 text-right" data-label="Days">{phase.days}</td><td className={`py-2 px-3 text-right ${metricTone(phase.total_return_pct)}`} data-label="Return">{signedPercent(phase.total_return_pct)}</td><td className="py-2 px-3 text-right" data-label="Sharpe">{format(phase.sharpe, 2)}</td><td className="py-2 px-3 text-right" data-label="Sortino">{format(phase.sortino, 2)}</td><td className="py-2 px-3 text-right text-red" data-label="Max DD">{isMetric(phase.max_dd_pct) ? `-${format(Math.abs(phase.max_dd_pct))}%` : '--'}</td><td className="py-2 px-3 text-right" data-label="Vol">{isMetric(phase.ann_vol_pct) ? `${format(phase.ann_vol_pct)}%` : '--'}</td><td className="py-2 px-3 text-right text-green" data-label="Best Day">{signedPercent(phase.best_day_pct)}</td><td className="py-2 px-3 text-right text-red" data-label="Worst Day">{signedPercent(phase.worst_day_pct)}</td><td className="py-2 px-3 text-right" data-label="Up / Down">{phase.up_days} / {phase.down_days}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      ) : null}

      <div className="stats-grid-four">
        <StatCard label="Avg Daily Return" value={signedPercent(metrics.avg_daily_pct, 3)} color={metricTone(metrics.avg_daily_pct)} />
        <StatCard label="Up / Down Days" value={`${metrics.up_days} / ${metrics.down_days}`} sub={`${((metrics.up_days / Math.max(metrics.up_days + metrics.down_days, 1)) * 100).toFixed(0)}% up days`} />
        <StatCard label="Best Day" value={signedPercent(metrics.best_day_pct)} color={isMetric(metrics.best_day_pct) ? 'text-green' : 'text-silver-dim'} />
        <StatCard label="Worst Day" value={signedPercent(metrics.worst_day_pct)} color={isMetric(metrics.worst_day_pct) ? 'text-red' : 'text-silver-dim'} />
      </div>

      {data.fun_facts[0] ? (
        <section className="research-note">
          <Lightbulb size={17} aria-hidden="true" />
          <div><p className="label">Research Note</p><p>{data.fun_facts[0]}</p></div>
        </section>
      ) : null}

      <p className="text-xs text-silver-dim text-center">Data sourced from Alpaca Markets portfolio history. Risk-free rate: 4.0%. Industry benchmarks are approximate public long-term averages.</p>
      <RetryNote error={analyticsQuery.error ?? data.error ?? null} hasData />
    </PageShell>
  );
}
