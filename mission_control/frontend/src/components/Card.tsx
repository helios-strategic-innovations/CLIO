import type { ReactNode } from 'react';

interface CardProps {
  title?: string;
  children: ReactNode;
  className?: string;
  action?: ReactNode;
}

export default function Card({ title, children, className = '', action }: CardProps) {
  return (
    <section className={`institutional-card ${className}`}>
      {title && (
        <header className="institutional-card-header">
          <h3 className="panel-title">{title}</h3>
          {action}
        </header>
      )}
      <div className="institutional-card-body">{children}</div>
    </section>
  );
}

export function StatCard({ label, value, sub, color }: { label: string; value: string; sub?: string; color?: string }) {
  return (
    <article className="institutional-card stat-card">
      <p className="label">{label}</p>
      <p className={`metric-value ${color || 'text-silver-light'}`}>{value}</p>
      {sub && <p className="stat-subline">{sub}</p>}
    </article>
  );
}
