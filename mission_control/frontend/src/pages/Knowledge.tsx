import { useMemo, useState } from 'react';
import { RefreshCw, X } from 'lucide-react';
import { usePolling } from '../hooks/usePolling';
import Card, { StatCard } from '../components/Card';
import PageShell from '../components/PageShell';
import { EmptyState, PanelSkeleton, RetryNote } from '../components/DataState';
import { fmtDate } from '../lib/format';

interface KnowledgeStats {
  total_cycles?: number;
  last_cycle?: string;
}

interface KnowledgeEntry {
  id: string;
  title: string;
  category?: string;
  created?: string;
  quality_score?: number;
  source?: string;
  tags?: string[];
  content?: string;
}

interface EntriesResponse {
  entries: KnowledgeEntry[];
  total?: number;
}

interface ContentResponse {
  content?: string;
}

export default function Knowledge() {
  const statsQuery = usePolling<KnowledgeStats>('/api/knowledge/stats', 60000);
  const entriesQuery = usePolling<EntriesResponse>('/api/knowledge?limit=50', 60000);
  const perspectiveQuery = usePolling<ContentResponse>('/api/knowledge/summary', 60000);
  const priorsQuery = usePolling<ContentResponse>('/api/knowledge/priors', 60000);
  const [selectedEntryId, setSelectedEntryId] = useState<string | null>(null);
  const [categoryFilter, setCategoryFilter] = useState('');
  const selectedQuery = usePolling<KnowledgeEntry>(selectedEntryId ? `/api/knowledge/${selectedEntryId}` : '/api/knowledge/__none__', 60000, Boolean(selectedEntryId));

  const items = useMemo(() => entriesQuery.data?.entries ?? [], [entriesQuery.data?.entries]);
  const categories = useMemo(() => [...new Set(items.map(entry => entry.category).filter((category): category is string => Boolean(category)))].sort(), [items]);
  const filtered = categoryFilter ? items.filter(entry => entry.category === categoryFilter) : items;
  const selectedEntry = selectedQuery.data;

  return (
    <PageShell className="space-y-6">
      <div className="page-heading">
        <p className="page-kicker">Knowledge</p>
        <h1 className="page-title">Knowledge Base</h1>
        <p className="page-subtitle">Hydrinia's living perspective, operating priors, and indexed memory corpus.</p>
      </div>

      <div className="stats-grid-four">
        <StatCard label="Total Entries" value={String(entriesQuery.data?.total ?? items.length)} />
        <StatCard label="Learning Cycles" value={String(statsQuery.data?.total_cycles ?? 0)} />
        <StatCard label="Categories" value={String(categories.length)} />
        <StatCard label="Last Cycle" value={fmtDate(statsQuery.data?.last_cycle)} />
      </div>

      <div className="section-grid-two">
        <div className="space-y-4">
          <Card title="Living Perspective" action={<button type="button" className="icon-button" title="Refresh perspective" onClick={perspectiveQuery.refetch}><RefreshCw size={15} aria-hidden="true" /><span className="sr-only">Refresh perspective</span></button>}>
            {perspectiveQuery.loading && !perspectiveQuery.data ? <PanelSkeleton /> : perspectiveQuery.data?.content ? <pre className="knowledge-content">{perspectiveQuery.data.content}</pre> : <EmptyState>No perspective available</EmptyState>}
            <RetryNote error={perspectiveQuery.error} hasData={Boolean(perspectiveQuery.data?.content)} />
          </Card>

          <Card title="Current Priors" action={<button type="button" className="icon-button" title="Refresh priors" onClick={priorsQuery.refetch}><RefreshCw size={15} aria-hidden="true" /><span className="sr-only">Refresh priors</span></button>}>
            {priorsQuery.loading && !priorsQuery.data ? <PanelSkeleton /> : priorsQuery.data?.content ? <pre className="knowledge-content max-h-64">{priorsQuery.data.content}</pre> : <EmptyState>No priors set</EmptyState>}
            <RetryNote error={priorsQuery.error} hasData={Boolean(priorsQuery.data?.content)} />
          </Card>
        </div>

        <Card title={`Entries (${filtered.length})`} action={
          <select value={categoryFilter} onChange={event => setCategoryFilter(event.target.value)} className="form-control compact-control" aria-label="Filter entries by category">
            <option value="">All Categories</option>
            {categories.map(category => <option key={category} value={category}>{category}</option>)}
          </select>
        }>
          {entriesQuery.loading && !entriesQuery.data ? <PanelSkeleton /> : filtered.length ? (
            <div className="knowledge-list">
              {filtered.map(entry => (
                <button key={entry.id} type="button" onClick={() => setSelectedEntryId(entry.id)} className={selectedEntryId === entry.id ? 'knowledge-row knowledge-row-active' : 'knowledge-row'}>
                  <span className="flex justify-between items-center gap-3"><strong>{entry.title}</strong><span className="quiet-meta shrink-0">{entry.category ?? 'Uncategorized'}</span></span>
                  <span className="flex justify-between text-xs text-silver-dim mt-1"><span>{fmtDate(entry.created)}</span>{entry.quality_score != null ? <span>Quality {entry.quality_score}/10</span> : null}</span>
                </button>
              ))}
            </div>
          ) : <EmptyState>No entries in this category</EmptyState>}
          <RetryNote error={entriesQuery.error} hasData={items.length > 0} />
        </Card>
      </div>

      {selectedEntryId ? (
        <Card title={selectedEntry?.title ?? 'Knowledge Entry'} action={<button type="button" className="icon-button" title="Close entry" onClick={() => setSelectedEntryId(null)}><X size={15} aria-hidden="true" /><span className="sr-only">Close entry</span></button>}>
          {selectedQuery.loading && !selectedEntry ? <PanelSkeleton /> : selectedEntry ? (
            <div className="space-y-3">
              <p className="quiet-meta">Category: {selectedEntry.category ?? 'Uncategorized'} / Source: {selectedEntry.source ?? 'Unavailable'} / Created: {fmtDate(selectedEntry.created)}{selectedEntry.quality_score != null ? ` / Quality: ${selectedEntry.quality_score}/10` : ''}</p>
              {selectedEntry.tags?.length ? <p className="text-xs text-silver-dim">Tags: {selectedEntry.tags.join(', ')}</p> : null}
              <pre className="knowledge-content">{selectedEntry.content || 'No content'}</pre>
            </div>
          ) : <EmptyState>Entry unavailable</EmptyState>}
          <RetryNote error={selectedQuery.error} hasData={Boolean(selectedEntry)} />
        </Card>
      ) : null}
    </PageShell>
  );
}
