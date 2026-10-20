export function fmtMoney(v: number | null | undefined, decimals = 2): string {
  if (v == null) return '--';
  return v.toLocaleString('en-US', { style: 'currency', currency: 'USD', minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export function fmtPct(v: number | null | undefined, decimals = 2): string {
  if (v == null) return '--';
  const sign = v >= 0 ? '+' : '';
  return `${sign}${v.toFixed(decimals)}%`;
}

export function fmtNum(v: number | null | undefined, decimals = 0): string {
  if (v == null) return '--';
  return v.toLocaleString('en-US', { minimumFractionDigits: decimals, maximumFractionDigits: decimals });
}

export function fmtDate(v: string | null | undefined): string {
  if (!v) return '--';
  try { return new Date(v).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); }
  catch { return v; }
}

export function plColor(v: number | null | undefined): string {
  if (v == null || v === 0) return 'text-silver-dim';
  return v > 0 ? 'text-green' : 'text-red';
}
