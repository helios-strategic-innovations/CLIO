import { useCallback, useEffect, useMemo, useRef, useState, type FormEvent } from 'react';
import { Check, Search, Send, XCircle } from 'lucide-react';
import { Area, AreaChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { usePolling } from '../hooks/usePolling';
import { useWebSocket } from '../hooks/useWebSocket';
import Card from '../components/Card';
import PageShell from '../components/PageShell';
import Signal from '../components/Signal';
import { EmptyState, Freshness, PanelSkeleton, RetryNote } from '../components/DataState';
import { fmtMoney, fmtNum, fmtPct, fmtDate, plColor } from '../lib/format';
import { post, del, get } from '../lib/api';

type OrderMode = 'equity' | 'options';
type ChartRange = '1D' | '5D' | '1M' | '3M' | '1Y';

interface PriceMessage {
  type: string;
  symbol: string;
  bid?: number;
  ask?: number;
  mid?: number;
  last?: number;
  price?: number;
  source?: string;
}

interface Quote {
  symbol: string;
  price: number | null;
  mark_price?: number | null;
  display_price?: number | null;
  change: number | null;
  change_pct: number | null;
  volume: number | null;
  bid: number | null;
  ask: number | null;
  beta: number | null;
  week52_high: number | null;
  week52_low: number | null;
  week52_source?: string | null;
  as_of?: string;
  source?: string;
  stale?: boolean;
}

interface CandlePoint {
  t: number | string;
  o: number;
  h: number;
  l: number;
  c: number;
  v: number;
}

interface Candles {
  symbol: string;
  range: ChartRange;
  timeframe: string;
  points: CandlePoint[];
  stale: boolean;
  as_of?: string;
  source?: string;
}

interface Order {
  id: string;
  submitted_at?: string;
  created_at?: string;
  symbol?: string;
  occ_symbol?: string;
  side?: string;
  qty?: number | string;
  type?: string;
  order_type?: string;
  limit_price?: number | string | null;
  status?: string;
  filled_qty?: number | string;
}

interface OptionContract {
  occ_symbol: string;
  expiration?: string;
  strike?: number;
  option_type?: string;
  bid?: number;
  ask?: number;
  mid?: number;
  implied_volatility?: number;
  delta?: number;
  theta?: number;
}

const CHART_RANGES: ChartRange[] = ['1D', '5D', '1M', '3M', '1Y'];
const EMPTY_VALUE = '\u2014';

function finite(value: unknown): number | null {
  const number = typeof value === 'number' ? value : Number(value);
  return Number.isFinite(number) ? number : null;
}

function optionLabel(symbol: string): string {
  const match = symbol.match(/^([A-Z]{1,6})(\d{2})(\d{2})(\d{2})([CP])(\d{8})$/);
  if (!match) return symbol;
  const [, root, , month, day, type, strikeRaw] = match;
  const strike = parseInt(strikeRaw, 10) / 1000;
  const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
  return `${root} ${months[parseInt(month, 10) - 1]} ${day} $${strike % 1 === 0 ? strike.toFixed(0) : strike.toFixed(2)} ${type === 'C' ? 'Call' : 'Put'}`;
}

function chartTick(value: number | string, range: ChartRange): string {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return range === '1D'
    ? date.toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })
    : date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
}

export default function Orders() {
  const ordersQuery = usePolling<{ orders: Order[]; as_of?: string; stale?: boolean }>('/api/orders', 10000);
  const [mode, setMode] = useState<OrderMode>('equity');
  const [range, setRange] = useState<ChartRange>('1D');
  const [equityForm, setEquityForm] = useState({ symbol: 'SPY', side: 'buy', qty: '', order_type: 'market', limit_price: '', time_in_force: 'day' });
  const [optionForm, setOptionForm] = useState({ underlying: 'SPY', occ_symbol: '', side: 'buy', qty: '1', position_intent: 'buy_to_open', time_in_force: 'day', limit_price_override: '' });
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<{ ok: boolean; message: string } | null>(null);
  const [liveQuotes, setLiveQuotes] = useState<Record<string, PriceMessage>>({});
  const [activeSymbol, setActiveSymbol] = useState('');
  const [chain, setChain] = useState<OptionContract[]>([]);
  const [chainLoading, setChainLoading] = useState(false);

  const optionRoot = optionForm.occ_symbol.match(/^([A-Z]{1,6})\d{6}[CP]\d{8}$/)?.[1] ?? '';
  const typedSymbol = (
    mode === 'equity' ? equityForm.symbol : optionRoot || optionForm.underlying
  ).trim().toUpperCase();
  useEffect(() => {
    const timer = window.setTimeout(() => setActiveSymbol(typedSymbol), 350);
    return () => window.clearTimeout(timer);
  }, [typedSymbol]);

  const onPriceMessage = useCallback((message: PriceMessage) => {
    if (message.symbol) setLiveQuotes(previous => ({ ...previous, [message.symbol]: message }));
  }, []);
  const { send: wsSend, connected: wsConnected } = useWebSocket<PriceMessage>('/ws/prices', onPriceMessage);
  const previousSymbol = useRef('');

  useEffect(() => {
    if (previousSymbol.current && previousSymbol.current !== activeSymbol && wsConnected) {
      wsSend({ unsubscribe: [previousSymbol.current] });
    }
    if (activeSymbol.length >= 2 && wsConnected) wsSend({ subscribe: [activeSymbol] });
    previousSymbol.current = activeSymbol;
  }, [activeSymbol, wsConnected, wsSend]);

  const quotePath = activeSymbol ? `/api/quote/${encodeURIComponent(activeSymbol)}/details` : '/api/quote/__none__/details';
  const candlesPath = activeSymbol
    ? `/api/quote/${encodeURIComponent(activeSymbol)}/candles?range=${range}`
    : '/api/quote/__none__/candles?range=1D';
  const quoteQuery = usePolling<Quote>(quotePath, 15000, activeSymbol.length >= 2);
  const candlesQuery = usePolling<Candles>(candlesPath, range === '1D' || range === '5D' ? 30000 : 3600000, activeSymbol.length >= 2);
  const quote = quoteQuery.data;
  const liveQuote = liveQuotes[activeSymbol];

  const mergedQuote = useMemo(() => ({
    price: finite(liveQuote?.last ?? liveQuote?.price ?? liveQuote?.mid ?? quote?.display_price ?? quote?.mark_price ?? quote?.price),
    change: finite(quote?.change),
    changePct: finite(quote?.change_pct),
    volume: finite(quote?.volume),
    bid: finite(liveQuote?.bid ?? quote?.bid),
    ask: finite(liveQuote?.ask ?? quote?.ask),
    beta: finite(quote?.beta),
    week52High: finite(quote?.week52_high),
    week52Low: finite(quote?.week52_low),
  }), [liveQuote, quote]);

  const submitEquity = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setResult(null);
    try {
      const body: Record<string, string | number> = {
        symbol: equityForm.symbol.toUpperCase(),
        side: equityForm.side,
        qty: parseFloat(equityForm.qty),
        order_type: equityForm.order_type,
        time_in_force: equityForm.time_in_force,
      };
      if (['limit', 'stop_limit'].includes(equityForm.order_type) && equityForm.limit_price) body.limit_price = parseFloat(equityForm.limit_price);
      await post('/api/orders', body);
      setResult({ ok: true, message: `Order submitted: ${body.side.toString().toUpperCase()} ${body.qty} ${body.symbol}` });
      setEquityForm(previous => ({ ...previous, qty: '', limit_price: '' }));
      ordersQuery.refetch();
    } catch (error) {
      setResult({ ok: false, message: error instanceof Error ? error.message : 'Order submission failed' });
    } finally {
      setSubmitting(false);
    }
  };

  const submitOption = async (event: FormEvent) => {
    event.preventDefault();
    setSubmitting(true);
    setResult(null);
    try {
      const body: Record<string, string | number> = {
        occ_symbol: optionForm.occ_symbol.toUpperCase(),
        side: optionForm.side,
        qty: parseInt(optionForm.qty, 10),
        position_intent: optionForm.position_intent,
        time_in_force: optionForm.time_in_force,
      };
      if (optionForm.limit_price_override) body.limit_price = parseFloat(optionForm.limit_price_override);
      await post('/api/options/orders', body);
      setResult({ ok: true, message: `Options order submitted: ${body.side.toString().toUpperCase()} ${body.qty} ${body.occ_symbol}` });
      setOptionForm(previous => ({ ...previous, qty: '1', limit_price_override: '' }));
      ordersQuery.refetch();
    } catch (error) {
      setResult({ ok: false, message: error instanceof Error ? error.message : 'Options order submission failed' });
    } finally {
      setSubmitting(false);
    }
  };

  const loadChain = async () => {
    if (!optionForm.underlying) return;
    setChainLoading(true);
    try {
      const response = await get<{ contracts: OptionContract[] }>(`/api/options/chain/${optionForm.underlying.toUpperCase()}?limit=30`);
      setChain(response.contracts ?? []);
    } catch {
      setChain([]);
    } finally {
      setChainLoading(false);
    }
  };

  const selectContract = (symbol: string) => {
    setOptionForm(previous => ({ ...previous, occ_symbol: symbol }));
    if (wsConnected) wsSend({ subscribe: [symbol] });
  };

  const cancelOrder = async (id: string) => {
    try {
      await del(`/api/orders/${id}`);
      ordersQuery.refetch();
    } catch {
      setResult({ ok: false, message: 'Order cancellation failed' });
    }
  };

  const orders = ordersQuery.data?.orders ?? [];
  const points = candlesQuery.data?.points ?? [];
  const formControl = 'form-control';
  const chartLabel = activeSymbol;

  return (
    <PageShell className="space-y-6">
      <div className="page-heading-row">
        <div className="page-heading">
          <p className="page-kicker">Orders</p>
          <h1 className="page-title">Execution Workbench</h1>
          <p className="page-subtitle">Guarded order entry with structured market context. Execution remains operator-controlled.</p>
        </div>
        <div className="view-tabs" role="tablist" aria-label="Order type">
          {(['equity', 'options'] as OrderMode[]).map(value => (
            <button
              key={value}
              type="button"
              role="tab"
              aria-selected={mode === value}
              aria-pressed={mode === value}
              onClick={() => { setMode(value); setResult(null); }}
            >
              {value}
            </button>
          ))}
        </div>
      </div>

      <Card
        title={activeSymbol ? `${chartLabel} Market Context` : 'Market Context'}
        action={activeSymbol ? <Freshness stale={quote?.stale || quoteQuery.isStale} asOf={quote?.as_of} source={quote?.source ?? liveQuote?.source} /> : undefined}
      >
        {!activeSymbol ? <EmptyState>Market context unavailable</EmptyState> : quoteQuery.loading && !quote ? <PanelSkeleton /> : (
          <>
            {mode === 'options' && optionForm.occ_symbol ? <p className="instrument-code mb-3" title={optionForm.occ_symbol}>{optionLabel(optionForm.occ_symbol)} · {optionForm.occ_symbol}</p> : null}
            <dl className="quick-stats">
              <div className="quick-stat"><dt>Price</dt><dd>{mergedQuote.price == null ? EMPTY_VALUE : fmtMoney(mergedQuote.price)}</dd></div>
              <div className="quick-stat"><dt>Chg</dt><dd className={plColor(mergedQuote.change)}>{mergedQuote.change == null ? EMPTY_VALUE : fmtMoney(mergedQuote.change)}</dd></div>
              <div className="quick-stat"><dt>% Chg</dt><dd className={plColor(mergedQuote.changePct)}>{mergedQuote.changePct == null ? EMPTY_VALUE : fmtPct(mergedQuote.changePct)}</dd></div>
              <div className="quick-stat"><dt>Volume</dt><dd>{mergedQuote.volume == null ? EMPTY_VALUE : fmtNum(mergedQuote.volume)}</dd></div>
              <div className="quick-stat"><dt>Bid</dt><dd>{mergedQuote.bid == null ? EMPTY_VALUE : fmtMoney(mergedQuote.bid)}</dd></div>
              <div className="quick-stat"><dt>Ask</dt><dd>{mergedQuote.ask == null ? EMPTY_VALUE : fmtMoney(mergedQuote.ask)}</dd></div>
              <div className="quick-stat"><dt>Beta</dt><dd>{mergedQuote.beta == null ? EMPTY_VALUE : mergedQuote.beta.toFixed(2)}</dd></div>
              <div className="quick-stat"><dt>52wk H / L</dt><dd title={quote?.week52_source ?? undefined}>{mergedQuote.week52High == null || mergedQuote.week52Low == null ? EMPTY_VALUE : `${fmtMoney(mergedQuote.week52High)} / ${fmtMoney(mergedQuote.week52Low)}`}</dd></div>
            </dl>
            <RetryNote error={quoteQuery.error} hasData={Boolean(quote)} />
          </>
        )}

        <div className="chart-meta mt-5">
          <div className="range-tabs" aria-label="Chart range">
            {CHART_RANGES.map(value => (
              <button key={value} type="button" aria-pressed={range === value} onClick={() => setRange(value)}>{value}</button>
            ))}
          </div>
          {candlesQuery.data ? <Freshness stale={candlesQuery.data.stale || candlesQuery.isStale} asOf={candlesQuery.data.as_of} source={candlesQuery.data.source} /> : null}
        </div>

        {!activeSymbol ? null : candlesQuery.loading && !candlesQuery.data ? <PanelSkeleton /> : points.length ? (
          <div className="chart-frame" aria-label={`${activeSymbol} ${range} price chart`}>
            <ResponsiveContainer width="100%" height="100%" minWidth={0} initialDimension={{ width: 1, height: 1 }}>
              <AreaChart data={points} margin={{ top: 8, right: 8, left: 2, bottom: 0 }}>
                <XAxis dataKey="t" tickFormatter={value => chartTick(value, range)} tick={{ fill: '#9f998f', fontSize: 11 }} axisLine={{ stroke: 'rgba(232,220,186,0.16)' }} tickLine={false} minTickGap={28} />
                <YAxis tickFormatter={value => `$${Number(value).toFixed(Number(value) < 100 ? 2 : 0)}`} tick={{ fill: '#9f998f', fontSize: 11 }} axisLine={false} tickLine={false} domain={['auto', 'auto']} width={58} />
                <Tooltip
                  contentStyle={{ background: '#201f1d', border: '1px solid rgba(232,220,186,0.3)', borderRadius: 4, color: '#eeeae2', fontSize: 13 }}
                  labelFormatter={value => new Date(value).toLocaleString()}
                  formatter={value => [fmtMoney(Number(value)), 'Close']}
                />
                <Area type="monotone" dataKey="c" stroke="#d6b867" strokeWidth={2} fill="#d6b867" fillOpacity={0.08} dot={false} isAnimationActive={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : <EmptyState>No chart points for this range</EmptyState>}
        <RetryNote error={candlesQuery.error} hasData={points.length > 0} />
      </Card>

      {mode === 'equity' ? (
        <Card title="New Equity Order">
          <form onSubmit={submitEquity} className="form-grid">
            <div className="form-field"><label className="label">Symbol</label><input value={equityForm.symbol} onChange={event => setEquityForm(previous => ({ ...previous, symbol: event.target.value }))} className={formControl} placeholder="SPY" required /></div>
            <div className="form-field"><label className="label">Side</label><select value={equityForm.side} onChange={event => setEquityForm(previous => ({ ...previous, side: event.target.value }))} className={formControl}><option value="buy">Buy</option><option value="sell">Sell</option></select></div>
            <div className="form-field"><label className="label">Quantity</label><input type="number" value={equityForm.qty} onChange={event => setEquityForm(previous => ({ ...previous, qty: event.target.value }))} className={formControl} placeholder="10" min="1" required /></div>
            <div className="form-field"><label className="label">Type</label><select value={equityForm.order_type} onChange={event => setEquityForm(previous => ({ ...previous, order_type: event.target.value }))} className={formControl}><option value="market">Market</option><option value="limit">Limit</option><option value="stop">Stop</option><option value="stop_limit">Stop Limit</option></select></div>
            {['limit', 'stop_limit'].includes(equityForm.order_type) ? <div className="form-field"><label className="label">Limit Price</label><input type="number" step="0.01" value={equityForm.limit_price} onChange={event => setEquityForm(previous => ({ ...previous, limit_price: event.target.value }))} className={formControl} required /></div> : null}
            <div className="form-field"><label className="label">TIF</label><select value={equityForm.time_in_force} onChange={event => setEquityForm(previous => ({ ...previous, time_in_force: event.target.value }))} className={formControl}><option value="day">Day</option><option value="gtc">GTC</option><option value="ioc">IOC</option></select></div>
            <button type="submit" disabled={submitting} className="primary-button"><Send size={15} aria-hidden="true" /> {submitting ? 'Submitting' : 'Submit'}</button>
          </form>
        </Card>
      ) : (
        <div className="section-grid-two">
          <Card title="Options Chain">
            <div className="flex flex-wrap items-end gap-3 mb-4">
              <div className="form-field flex-1 min-w-32"><label className="label">Underlying</label><input value={optionForm.underlying} onChange={event => setOptionForm(previous => ({ ...previous, underlying: event.target.value }))} className={formControl} placeholder="SPY" /></div>
              <button type="button" onClick={loadChain} disabled={chainLoading || !optionForm.underlying} className="secondary-button"><Search size={15} aria-hidden="true" /> {chainLoading ? 'Loading' : 'Load Chain'}</button>
            </div>
            {chain.length ? (
              <div className="table-scroll max-h-96 overflow-y-auto">
                <table>
                  <thead><tr><th className="py-2 px-2 text-left">Exp</th><th className="py-2 px-2 text-right">Strike</th><th className="py-2 px-2 text-left">Type</th><th className="py-2 px-2 text-right">Bid</th><th className="py-2 px-2 text-right">Ask</th><th className="py-2 px-2 text-right">Mid</th><th className="py-2 px-2 text-right">IV</th><th className="py-2 px-2 text-right">Delta</th><th className="py-2 px-2 text-right">Theta</th><th><span className="sr-only">Select</span></th></tr></thead>
                  <tbody>
                    {chain.map(contract => (
                      <tr key={contract.occ_symbol}>
                        <td className="py-2 px-2">{contract.expiration ?? EMPTY_VALUE}</td><td className="py-2 px-2 text-right">{fmtMoney(contract.strike)}</td><td className="py-2 px-2">{contract.option_type?.toLowerCase().includes('call') ? 'Call' : 'Put'}</td><td className="py-2 px-2 text-right">{fmtMoney(contract.bid)}</td><td className="py-2 px-2 text-right">{fmtMoney(contract.ask)}</td><td className="py-2 px-2 text-right">{fmtMoney(contract.mid)}</td><td className="py-2 px-2 text-right">{contract.implied_volatility == null ? EMPTY_VALUE : `${(contract.implied_volatility * 100).toFixed(1)}%`}</td><td className="py-2 px-2 text-right">{contract.delta?.toFixed(3) ?? EMPTY_VALUE}</td><td className="py-2 px-2 text-right">{contract.theta?.toFixed(3) ?? EMPTY_VALUE}</td>
                        <td className="py-2 px-2 text-right"><button type="button" className="icon-button" title={`Select ${contract.occ_symbol}`} onClick={() => selectContract(contract.occ_symbol)}><Check size={15} aria-hidden="true" /><span className="sr-only">Select contract</span></button></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : <EmptyState>{chainLoading ? 'Loading option contracts' : 'No option contracts loaded'}</EmptyState>}
          </Card>

          <Card title="New Options Order">
            <form onSubmit={submitOption} className="form-grid">
              <div className="form-field form-span-2"><label className="label">OCC Symbol</label><input value={optionForm.occ_symbol} onChange={event => setOptionForm(previous => ({ ...previous, occ_symbol: event.target.value }))} className={formControl} placeholder="SPY250321C00590000" required /></div>
              <div className="form-field"><label className="label">Side</label><select value={optionForm.side} onChange={event => setOptionForm(previous => ({ ...previous, side: event.target.value }))} className={formControl}><option value="buy">Buy</option><option value="sell">Sell</option></select></div>
              <div className="form-field"><label className="label">Contracts</label><input type="number" value={optionForm.qty} onChange={event => setOptionForm(previous => ({ ...previous, qty: event.target.value }))} className={formControl} min="1" max="100" required /></div>
              <div className="form-field"><label className="label">Intent</label><select value={optionForm.position_intent} onChange={event => setOptionForm(previous => ({ ...previous, position_intent: event.target.value }))} className={formControl}><option value="buy_to_open">Buy to Open</option><option value="buy_to_close">Buy to Close</option><option value="sell_to_open">Sell to Open</option><option value="sell_to_close">Sell to Close</option></select></div>
              <div className="form-field"><label className="label">Limit Override</label><input type="number" step="0.01" value={optionForm.limit_price_override} onChange={event => setOptionForm(previous => ({ ...previous, limit_price_override: event.target.value }))} className={formControl} placeholder="Auto mid" /></div>
              <button type="submit" disabled={submitting} className="primary-button"><Send size={15} aria-hidden="true" /> {submitting ? 'Submitting' : 'Submit'}</button>
            </form>
          </Card>
        </div>
      )}

      {result ? <p className={`inline-result ${result.ok ? 'text-buy' : 'text-sell'}`} role="status">{result.message}</p> : null}

      <Card title={`Active Orders (${orders.length})`} action={<Freshness stale={ordersQuery.data?.stale || ordersQuery.isStale} asOf={ordersQuery.data?.as_of} />}>
        {ordersQuery.loading && !ordersQuery.data ? <PanelSkeleton /> : orders.length ? (
          <div className="table-scroll">
            <table className="mobile-table">
              <thead><tr><th className="py-2 px-2 text-left">Time</th><th className="py-2 px-2 text-left">Symbol</th><th className="py-2 px-2 text-left">Side</th><th className="py-2 px-2 text-right">Qty</th><th className="py-2 px-2 text-left">Type</th><th className="py-2 px-2 text-right">Limit</th><th className="py-2 px-2 text-left">Status</th><th className="py-2 px-2 text-right">Filled</th><th><span className="sr-only">Actions</span></th></tr></thead>
              <tbody>
                {orders.map(order => {
                  const symbol = order.symbol ?? order.occ_symbol ?? EMPTY_VALUE;
                  return (
                    <tr key={order.id}>
                      <td className="py-2 px-2" data-label="Time">{fmtDate(order.submitted_at ?? order.created_at)}</td>
                      <td className="py-2 px-2" data-primary="true"><span className="instrument-label" title={symbol}>{optionLabel(symbol)}</span>{symbol !== optionLabel(symbol) ? <span className="instrument-code">{symbol}</span> : null}</td>
                      <td className="py-2 px-2" data-label="Side"><Signal value={order.side} /></td>
                      <td className="py-2 px-2 text-right" data-label="Qty">{order.qty ?? EMPTY_VALUE}</td>
                      <td className="py-2 px-2" data-label="Type">{order.type ?? order.order_type ?? EMPTY_VALUE}</td>
                      <td className="py-2 px-2 text-right" data-label="Limit">{finite(order.limit_price) == null ? EMPTY_VALUE : fmtMoney(finite(order.limit_price))}</td>
                      <td className="py-2 px-2" data-label="Status">{order.status ?? EMPTY_VALUE}</td>
                      <td className="py-2 px-2 text-right" data-label="Filled">{order.filled_qty ?? '0'}</td>
                      <td className="py-2 px-2 text-right" data-label="Action"><button type="button" className="icon-button" title="Cancel order" onClick={() => cancelOrder(order.id)}><XCircle size={16} aria-hidden="true" /><span className="sr-only">Cancel order</span></button></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : <EmptyState>No active orders</EmptyState>}
        <RetryNote error={ordersQuery.error} hasData={orders.length > 0} />
      </Card>
    </PageShell>
  );
}
