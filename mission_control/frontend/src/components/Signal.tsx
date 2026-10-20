type SignalTone = 'buy' | 'sell' | 'hold' | 'unknown';

const BUY_VALUES = new Set(['BUY', 'LONG', 'BULLISH', 'CALL']);
const SELL_VALUES = new Set(['SELL', 'SHORT', 'BEARISH', 'PUT']);
const HOLD_VALUES = new Set(['HOLD', 'NEUTRAL', 'WAIT', 'FLAT', 'MIXED', 'RISK_OFF']);

function normalizeSignal(value: unknown): { label: string; tone: SignalTone } {
  const normalized = String(value ?? '').trim().toUpperCase();
  if (BUY_VALUES.has(normalized) || normalized.includes('BULL') || normalized === 'RISK_ON') return { label: 'BUY', tone: 'buy' };
  if (SELL_VALUES.has(normalized) || normalized.includes('BEAR')) return { label: 'SELL', tone: 'sell' };
  if (HOLD_VALUES.has(normalized)) return { label: 'HOLD', tone: 'hold' };
  return { label: normalized || 'UNAVAILABLE', tone: 'unknown' };
}

export default function Signal({ value, className = '' }: { value: unknown; className?: string }) {
  const signal = normalizeSignal(value);
  return (
    <span className={`signal signal-${signal.tone} ${className}`} aria-label={`Signal: ${signal.label}`}>
      <span className="signal-dot" aria-hidden="true" />
      {signal.label}
    </span>
  );
}
