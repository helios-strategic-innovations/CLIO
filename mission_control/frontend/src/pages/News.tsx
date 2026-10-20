import { useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import Card from '../components/Card';
import PageShell from '../components/PageShell';
import { EmptyState, Freshness, PanelSkeleton, RetryNote } from '../components/DataState';
import { fmtDate } from '../lib/format';

interface NewsItem {
  id?: string;
  title?: string;
  headline?: string;
  category?: string;
  topic?: string;
  summary?: string;
  sent_at?: string;
  timestamp?: string;
  symbols?: string[];
}

interface NewsResponse {
  items: NewsItem[];
  as_of?: string;
  stale?: boolean;
}

interface BreakingResponse {
  breaking: NewsItem[];
  categories: string[];
  as_of?: string;
  stale?: boolean;
}

function CategoryLabel({ category }: { category: string }) {
  return <span className="quiet-meta shrink-0">{category}</span>;
}

function NewsRows({ items, detailed = false }: { items: NewsItem[]; detailed?: boolean }) {
  return (
    <div className="news-list">
      {items.map((item, index) => (
        <article key={item.id ?? `${item.title ?? item.headline ?? 'news'}-${index}`} className="news-row">
          <div className="flex justify-between items-start gap-3">
            <strong>{item.title ?? item.headline ?? 'Untitled'}</strong>
            <CategoryLabel category={item.category ?? item.topic ?? 'General'} />
          </div>
          {detailed && item.summary ? <p>{item.summary}</p> : null}
          <div className="news-meta">
            <time>{fmtDate(item.sent_at ?? item.timestamp)}</time>
            {item.symbols?.length ? <span>Symbols: {item.symbols.slice(0, 5).join(', ')}</span> : null}
          </div>
        </article>
      ))}
    </div>
  );
}

export default function News() {
  const newsQuery = usePolling<NewsResponse>('/api/news', 30000);
  const breakingQuery = usePolling<BreakingResponse>('/api/news/breaking?hours=12', 15000);
  const [categoryFilter, setCategoryFilter] = useState('');
  const breakingItems = breakingQuery.data?.breaking ?? [];
  const categories = breakingQuery.data?.categories ?? [];
  const filtered = categoryFilter ? breakingItems.filter(item => item.category === categoryFilter) : breakingItems;

  return (
    <PageShell className="space-y-6">
      <div className="page-heading">
        <p className="page-kicker">Intelligence</p>
        <h1 className="page-title">News &amp; Intelligence</h1>
        <p className="page-subtitle">Guarded market context and categorized breaking coverage.</p>
      </div>

      <div className="section-grid-two">
        <Card title="Breaking News" action={
          <div className="flex items-center gap-2">
            {breakingQuery.data ? <Freshness stale={breakingQuery.data.stale || breakingQuery.isStale} asOf={breakingQuery.data.as_of} /> : null}
            <select value={categoryFilter} onChange={event => setCategoryFilter(event.target.value)} className="form-control compact-control" aria-label="Filter breaking news by category">
              <option value="">All Categories</option>
              {categories.map(category => <option key={category} value={category}>{category}</option>)}
            </select>
          </div>
        }>
          {breakingQuery.loading && !breakingQuery.data ? <PanelSkeleton /> : filtered.length ? <NewsRows items={filtered} detailed /> : <EmptyState>{categoryFilter ? `No breaking news in ${categoryFilter}` : 'No breaking news'}</EmptyState>}
          <RetryNote error={breakingQuery.error} hasData={breakingItems.length > 0} />
        </Card>

        <Card title="Intelligence Feed" action={newsQuery.data ? <Freshness stale={newsQuery.data.stale || newsQuery.isStale} asOf={newsQuery.data.as_of} /> : undefined}>
          {newsQuery.loading && !newsQuery.data ? <PanelSkeleton /> : newsQuery.data?.items?.length ? <NewsRows items={newsQuery.data.items} /> : <EmptyState>No recent intelligence</EmptyState>}
          <RetryNote error={newsQuery.error} hasData={Boolean(newsQuery.data?.items?.length)} />
        </Card>
      </div>
    </PageShell>
  );
}
