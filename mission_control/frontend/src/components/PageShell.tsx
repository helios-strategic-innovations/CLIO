import type { ReactNode } from 'react';

export default function PageShell({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`page-shell ${className}`}>{children}</section>;
}
