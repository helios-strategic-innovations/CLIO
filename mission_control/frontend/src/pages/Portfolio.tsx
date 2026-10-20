import { useMemo, useState } from 'react';
import { RefreshCw } from 'lucide-react';
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { usePolling } from '../hooks/usePolling';
import Card, { StatCard } from '../components/Card';
import PageShell from '../components/PageShell';
import Signal from '../components/Signal';
import { EmptyState, Freshness, PanelSkeleton, RetryNote } from '../components/DataState';
import { fmtMoney, fmtPct, fmtNum, plColor, fmtDate } from '../lib/format';

type PortfolioRange = '1D' | '5D' | '1M' | '3M' | '1Y' | 'ALL';

interface HistoryPoint {
  t: number | string;
  equity: number | null;
}

interface HistoryResponse {
  points: HistoryPoint[];
  as_of?: string;
  source?: string;
  stale?: boolean;
}

interface Position {
  symbol: string;
  asset_class?: string;
  side?: string;
  qty: number;
  avg_entry_price: number;
  current_price: number;
  market_value: number;
  unrealized_pl: number;
  unrealized_plpc: number;
}

interface PortfolioResponse {
  account: {
    portfolio_value?: number;
    equity: number;
    cash: number;
    buying_power: number;
    day_pl: number;
    trading_blocked?: boolean;
    pattern_day_trader?: boolean;
  };
  positions: Position[];
  as_of?: string;
  stale?: boolean;
}

interface TaxLot {
  lot_id?: string;
  symbol: string;
  is_options?: boolean;
  contract?: string;
  status?: string;
  quantity?: number;
  entry_price?: number;
  estimated_cost_basis?: number;
  pnl?: number;
  holding_days?: number;
  wash_sale_risk?: boolean;
  holding_term?: string;
}

interface TaxLotsResponse {
  lots: TaxLot[];
  total_count?: number;
  open_count?: number;
  closed_count?: number;
  as_of?: string;
  stale?: boolean;
  source?: string;
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

const RANGE_CONTRACT: Record<PortfolioRange, { period: string; timeframe: string }> = {
  '1D': { period: '1D', timeframe: '15Min' },
  '5D': { period: '1W', timeframe: '1H' },
  '1M': { period: '1M', timeframe: '1D' },
  '3M': { period: '3M', timeframe: '1D' },
  '1Y': { period: '1A', timeframe: '1D' },
  'ALL': { period: 'all', timeframe: '1D' },
};

function optionLabel(symbol: string): string {
  const match = symbol.match(/^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/);
  if (!match) return symbol;
  const [, root, , month, day, type, strikeRaw] = match;
  const strike = parseInt(strikeRaw, 10) / 1000;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${root} ${months[parseInt(month, 10) - 1]} ${day} $${strike % 1 === 0 ? strike.toFixed(0) : strike.toFixed(2)} ${type === 'C' ? 'Call' : 'Put'}`;
}

function finite(value: unknown): number | null {
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

export default function Portfolio() {
  const portfolioQuery = usePolling<PortfolioResponse>('/api/portfolio', 15000);
  const taxLotsQuery = usePolling<TaxLotsResponse>('/api/portfolio/tax-lots?limit=80', 60000);
  const [range, setRange] = useState<PortfolioRange>('5D');
  const [tradesEnabled, setTradesEnabled] = useState(false);
  const contract = RANGE_CONTRACT[range];
  const historyQuery = usePolling<HistoryResponse>(`/api/portfolio/history?period=${contract.period}&timeframe=${contract.timeframe}`, 60000);
  const tradesQuery = usePolling<{ trades: Trade[]; as_of?: string; stale?: boolean }>('/api/trades?limit=50', 60000, tradesEnabled);

  const portfolio = portfolioQuery.data;
  const taxLots = taxLotsQuery.data;
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

  const points = useMemo(
    () => (historyQuery.data?.points ?? []).filter((point): point is HistoryPoint & { equity: number } => point.equity != null && Number.isFinite(Number(point.equity))),
    [historyQuery.data?.points],
  );

  const positionRows = (positions: Position[], isOption: boolean) => positions.map(position => (
    <tr key={position.symbol}>
      <td className="py-3 px-3" data-primary="true">
        <span className="instrument-label" title={isOption ? optionLabel(position.symbol) : position.symbol}>{isOption ? optionLabel(position.symbol) : position.symbol}</span>
        {isOption ? <span className="instrument-code" title={position.symbol}>{position.symbol}</span> : null}
      </td>
      <td className="py-3 px-3" data-label="Side"><Signal value={position.side} /></td>
      <td className="py-3 px-3 text-right" data-label="Qty">{fmtNum(Math.abs(position.qty))}</td>
      <td className="py-3 px-3 text-right" data-label="Entry">{fmtMoney(position.avg_entry_price)}</td>
      <td className="py-3 px-3 text-right" data-label="Current">{fmtMoney(position.current_price)}</td>
      <td className="py-3 px-3 text-right" data-label="Market Value">{fmtMoney(position.market_value)}</td>
      <td className={`py-3 px-3 text-right ${plColor(position.unrealized_pl)}`} data-label="Unrealized P&L">{fmtMoney(position.unrealized_pl)}</td>
      <td className={`py-3 px-3 text-right ${plColor(position.unrealized_plpc)}`} data-label="Return">{fmtPct(position.unrealized_plpc)}</td>
    </tr>
  ));

  const positionHeader = (
    <tr><th className="py-3 px-3 text-left">Instrument</th><th className="py-3 px-3 text-left">Side</th><th className="py-3 px-3 text-right">Qty</th><th className="py-3 px-3 text-right">Entry</th><th className="py-3 px-3 text-right">Current</th><th className="py-3 px-3 text-right">Market Value</th><th className="py-3 px-3 text-right">Unrealized P&amp;L</th><th className="py-3 px-3 text-right">Return</th></tr>
  );

  return (
    <PageShell className="space-y-6">
      <div className="page-heading-row">
        <div className="page-heading">
          <p className="page-kicker">Portfolio</p>
          <h1 className="page-title">Capital Ledger</h1>
          <p className="page-subtitle">Allocation, portfolio history, holdings, and Hydrinia-derived tax-lot estimates.</p>
        </div>
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <span className="quiet-meta">Account <strong className={account?.trading_blocked ? 'text-sell' : 'text-buy'}>{account?.trading_blocked ? 'Blocked' : 'Active'}</strong></span>
          <span className="quiet-meta">PDT <strong className="text-silver-light">{account?.pattern_day_trader ? 'Yes' : 'No'}</strong></span>
          <Freshness stale={portfolio?.stale || portfolioQuery.isStale} asOf={portfolio?.as_of} source="Portfolio" />
        </div>
      </div>

      <Card
        title="Equity Curve"
        action={
          <div className="range-tabs" aria-label="Portfolio history range">
            {(Object.keys(RANGE_CONTRACT) as PortfolioRange[]).map(value => <button key={value} type="button" aria-pressed={range === value} onClick={() => setRange(value)}>{value}</button>)}
          </div>
        }
      >
        <div className="chart-meta">
          <span className="chart-source">{contract.period} / {contract.timeframe}</span>
          {historyQuery.data ? <Freshness stale={historyQuery.data.stale || historyQuery.isStale} asOf={historyQuery.data.as_of} source={historyQuery.data.source} /> : null}
        </div>
        {historyQuery.loading && !historyQuery.data ? <PanelSkeleton /> : points.length ? (
          <div className="chart-frame" aria-label={`${range} portfolio equity chart`}>
            <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1, height: 1 }}>
              <LineChart data={points} margin={{ top: 8, right: 8, left: 2, bottom: 0 }}>
                <XAxis dataKey="t" tick={{ fill: '#9f998f', fontSize: 11 }} tickFormatter={value => new Date(value).toLocaleDateString('en-US', { month: 'short', day: 'numeric' })} axisLine={{ stroke: 'rgba(232,220,186,0.16)' }} tickLine={false} minTickGap={30} />
                <YAxis tick={{ fill: '#9f998f', fontSize: 11 }} tickFormatter={value => `$${(Number(value) / 1000).toFixed(0)}k`} axisLine={false} tickLine={false} domain={['auto', 'auto']} width={58} />
                <Tooltip contentStyle={{ background: '#201f1d', border: '1px solid rgba(232,220,186,0.3)', borderRadius: 4, color: '#eeeae2', fontSize: 13 }} labelFormatter={value => new Date(value).toLocaleString()} formatter={value => [fmtMoney(Number(value)), 'Equity']} />
                <Line type="monotone" dataKey="equity" stroke="#d6b867" dot={false} strokeWidth={2} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        ) : <EmptyState>No equity history for this period yet</EmptyState>}
        <RetryNote error={historyQuery.error} hasData={points.length > 0} />
      </Card>

      {portfolioQuery.loading && !portfolio ? <PanelSkeleton /> : (
        <div className="stats-grid-four">
          <StatCard label="Portfolio Value" value={fmtMoney(account?.portfolio_value ?? account?.equity)} />
          <StatCard label="Cash" value={fmtMoney(account?.cash)} />
          <StatCard label="Buying Power" value={fmtMoney(account?.buying_power)} />
          <StatCard label="Day P&L" value={fmtMoney(account?.day_pl)} color={plColor(account?.day_pl)} />
        </div>
      )}
      <RetryNote error={portfolioQuery.error} hasData={Boolean(portfolio)} />

      <Card title={`Equity Positions (${equities.length})`}>
        {equities.length ? <div className="table-scroll"><table className="mobile-table"><thead>{positionHeader}</thead><tbody>{positionRows(equities, false)}</tbody></table></div> : <EmptyState>No equity positions</EmptyState>}
      </Card>

      {options.length ? <Card title={`Option Positions (${options.length})`}><div className="table-scroll"><table className="mobile-table"><thead>{positionHeader}</thead><tbody>{positionRows(options, true)}</tbody></table></div></Card> : null}

      <Card title={`Estimated Tax Lots (${taxLots?.total_count ?? taxLots?.lots?.length ?? 0})`} action={taxLots ? <Freshness stale={taxLots.stale || taxLotsQuery.isStale} asOf={taxLots.as_of} source={taxLots.source} /> : undefined}>
        {taxLotsQuery.loading && !taxLots ? <PanelSkeleton /> : taxLots?.lots?.length ? (
          <>
            <div className="flex flex-wrap items-center gap-3 mb-3">
              <span className="estimate-chip">Estimate - Not Broker-Verified</span>
              <span className="quiet-meta">Hydrinia position log</span>
              <span className="quiet-meta">Open {taxLots.open_count ?? 0} / Closed {taxLots.closed_count ?? 0}</span>
            </div>
            <p className="text-xs text-silver-dim mb-3">Derived from Hydrinia position logs with broker_verified=false. Cost basis and wash-sale flags remain estimates until broker-confirmed lots are available.</p>
            <div className="table-scroll">
              <table className="mobile-table">
                <thead><tr><th className="py-3 px-3 text-left">Lot</th><th className="py-3 px-3 text-left">Status</th><th className="py-3 px-3 text-right">Qty</th><th className="py-3 px-3 text-right">Entry</th><th className="py-3 px-3 text-right">Cost Basis</th><th className="py-3 px-3 text-right">P&amp;L</th><th className="py-3 px-3 text-right">Days</th><th className="py-3 px-3 text-right">Tax</th></tr></thead>
                <tbody>
                  {taxLots.lots.map((lot, index) => {
                    const symbol = lot.is_options ? lot.contract ?? lot.symbol : lot.symbol;
                    return (
                      <tr key={lot.lot_id ?? `${lot.symbol}-${index}`}>
                        <td className="py-3 px-3" data-primary="true"><span className="instrument-label" title={optionLabel(symbol)}>{lot.is_options ? optionLabel(symbol) : symbol}</span>{lot.is_options ? <span className="instrument-code">{symbol}</span> : null}</td>
                        <td className="py-3 px-3" data-label="Status">{lot.status?.toUpperCase() ?? '--'}</td><td className="py-3 px-3 text-right" data-label="Qty">{fmtNum(Math.abs(lot.quantity ?? 0))}</td><td className="py-3 px-3 text-right" data-label="Entry">{fmtMoney(lot.entry_price)}</td><td className="py-3 px-3 text-right" data-label="Cost Basis">{fmtMoney(lot.estimated_cost_basis)}</td><td className={`py-3 px-3 text-right ${plColor(lot.pnl)}`} data-label="P&L">{fmtMoney(lot.pnl)}</td><td className="py-3 px-3 text-right" data-label="Days">{lot.holding_days ?? '--'}</td><td className={`py-3 px-3 text-right ${lot.wash_sale_risk ? 'text-hold' : ''}`} data-label="Tax">{lot.wash_sale_risk ? 'WATCH' : lot.holding_term === 'long_term' ? 'LT' : 'ST'}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        ) : <EmptyState>{taxLotsQuery.error ? 'Awaiting backend activation - estimates unavailable' : 'No Hydrinia position lots available'}</EmptyState>}
        <RetryNote error={taxLotsQuery.error} hasData={Boolean(taxLots?.lots?.length)} />
      </Card>

      <Card
        title="Recent Trades"
        action={<button type="button" className="secondary-button" onClick={() => tradesEnabled ? tradesQuery.refetch() : setTradesEnabled(true)} disabled={tradesQuery.isFetching}><RefreshCw size={15} aria-hidden="true" /> {tradesQuery.isFetching ? 'Refreshing' : tradesEnabled ? 'Refresh' : 'Load Trades'}</button>}
      >
        {!tradesEnabled ? <EmptyState>Load recent trades to view execution history</EmptyState> : tradesQuery.loading && !tradesQuery.data ? <PanelSkeleton /> : tradesQuery.data?.trades?.length ? (
          <div className="table-scroll"><table className="mobile-table"><thead><tr><th className="py-2 px-2 text-left">Time</th><th className="py-2 px-2 text-left">Symbol</th><th className="py-2 px-2 text-left">Side</th><th className="py-2 px-2 text-right">Qty</th><th className="py-2 px-2 text-right">Price</th><th className="py-2 px-2 text-left">Status</th></tr></thead><tbody>{tradesQuery.data.trades.map((trade, index) => <tr key={trade.id ?? index}><td className="py-2 px-2" data-label="Time">{fmtDate(trade.filled_at ?? trade.submitted_at)}</td><td className="py-2 px-2" data-primary="true"><span className="instrument-label">{optionLabel(trade.symbol)}</span></td><td className="py-2 px-2" data-label="Side"><Signal value={trade.side} /></td><td className="py-2 px-2 text-right" data-label="Qty">{finite(trade.filled_qty ?? trade.qty) ?? '--'}</td><td className="py-2 px-2 text-right" data-label="Price">{fmtMoney(finite(trade.filled_avg_price))}</td><td className="py-2 px-2" data-label="Status">{trade.status ?? '--'}</td></tr>)}</tbody></table></div>
        ) : <EmptyState>No recent trades</EmptyState>}
        <RetryNote error={tradesQuery.error} hasData={Boolean(tradesQuery.data?.trades?.length)} />
      </Card>
    </PageShell>
  );
}
