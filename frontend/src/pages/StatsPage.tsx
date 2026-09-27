import { useEffect, useState } from 'react';
import { getStats, rebuildIndex } from '../services/api';
import type { StatsResponse } from '../types/api';

export default function StatsPage() {
  const [stats, setStats] = useState<StatsResponse | null>(null);
  const [rebuilding, setRebuilding] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function refresh() {
    try {
      setStats(await getStats());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load stats');
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleRebuild() {
    setRebuilding(true);
    setError(null);
    try {
      await rebuildIndex();
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Rebuild failed');
    } finally {
      setRebuilding(false);
    }
  }

  if (!stats) return <div className="stats-page">{error ?? 'Loading...'}</div>;

  return (
    <div className="stats-page">
      <div className="stats-grid">
        <StatCard label="Documents indexed" value={stats.documents_indexed.toLocaleString()} />
        <StatCard label="Vocabulary size" value={stats.unique_terms.toLocaleString()} />
        <StatCard label="Total indexed tokens" value={stats.total_tokens.toLocaleString()} />
        <StatCard label="Average document length" value={`${stats.average_doc_length} tokens`} />
        <StatCard label="Index size" value={`${(stats.index_size_bytes / 1024).toFixed(1)} KB`} />
        <StatCard
          label="Latest search latency"
          value={stats.last_search_latency_ms != null ? `${stats.last_search_latency_ms.toFixed(2)}ms` : '—'}
        />
        <StatCard
          label="Last full rebuild"
          value={stats.last_index_build_ms != null ? `${stats.last_index_build_ms.toFixed(2)}ms` : '—'}
        />
        <StatCard label="Last rebuild time" value={stats.last_rebuild_ts ?? 'Never'} />
      </div>

      {error && <div className="error-banner">{error}</div>}

      <button className="rebuild-button" onClick={handleRebuild} disabled={rebuilding}>
        {rebuilding ? 'Rebuilding index...' : 'Rebuild Index'}
      </button>
    </div>
  );
}

function StatCard({ label, value }: { label: string; value: string }) {
  return (
    <div className="stat-card">
      <div className="stat-value">{value}</div>
      <div className="stat-label">{label}</div>
    </div>
  );
}
