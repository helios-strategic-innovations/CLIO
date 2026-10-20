import { CircleSlash2 } from 'lucide-react';

export function PanelSkeleton({ rows = 3 }: { rows?: number }) {
  return (
    <div className="panel-skeleton" aria-label="Loading data" role="status">
      {Array.from({ length: rows }, (_, index) => <span key={index} />)}
    </div>
  );
}

export function RetryNote({ error, hasData = false }: { error: string | null; hasData?: boolean }) {
  if (!error) return null;
  return (
    <p className="retry-note" role="status">
      {hasData ? 'Showing last good data; retrying...' : 'Data unavailable; retrying...'}
    </p>
  );
}

export function EmptyState({ children }: { children: string }) {
  return (
    <p className="empty-state">
      <CircleSlash2 size={15} aria-hidden="true" />
      <span>{children}</span>
    </p>
  );
}

export function Freshness({
  stale,
  asOf,
  source,
}: {
  stale?: boolean;
  asOf?: string | number | null;
  source?: string | null;
}) {
  const age = asOf ? new Date(asOf).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }) : null;
  return (
    <span className={`freshness-chip ${stale ? 'freshness-stale' : 'freshness-ok'}`} title={[source, age].filter(Boolean).join(' - ')}>
      {stale ? 'Stale' : 'Current'}
      {source ? <span>{source}</span> : null}
    </span>
  );
}
