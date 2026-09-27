import { useState, type FormEvent } from 'react';
import { search } from '../services/api';
import type { RankingStrategy, SearchResponse } from '../types/api';
import ResultCard from '../components/ResultCard';

export default function SearchPage() {
  const [query, setQuery] = useState('');
  const [ranking, setRanking] = useState<RankingStrategy>('tfidf');
  const [topK, setTopK] = useState(10);
  const [response, setResponse] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function runSearch(page = 1) {
    if (!query.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const res = await search({ query, ranking, top_k: topK, page });
      setResponse(res);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Search failed');
    } finally {
      setLoading(false);
    }
  }

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    runSearch(1);
  }

  const totalPages = response ? Math.max(1, Math.ceil(response.total_matches / response.top_k)) : 1;

  return (
    <div className="search-page">
      <form className="search-bar" onSubmit={handleSubmit}>
        <input
          type="text"
          placeholder="Search documents..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          autoFocus
        />
        <button type="submit" disabled={loading}>
          {loading ? 'Searching...' : 'Search'}
        </button>
      </form>

      <div className="search-controls">
        <div className="ranking-selector">
          <label>
            <input
              type="radio"
              name="ranking"
              checked={ranking === 'tfidf'}
              onChange={() => setRanking('tfidf')}
            />
            TF-IDF
          </label>
          <label>
            <input
              type="radio"
              name="ranking"
              checked={ranking === 'bm25'}
              onChange={() => setRanking('bm25')}
            />
            BM25
          </label>
        </div>
        <label className="topk-selector">
          Results per page:
          <select value={topK} onChange={(e) => setTopK(Number(e.target.value))}>
            <option value={5}>5</option>
            <option value={10}>10</option>
            <option value={20}>20</option>
          </select>
        </label>
      </div>

      {error && <div className="error-banner">{error}</div>}

      {response && (
        <div className="results-meta">
          {response.total_matches} result{response.total_matches !== 1 ? 's' : ''} for "
          {response.query}" &middot; {response.search_time_ms.toFixed(1)}ms &middot;{' '}
          {response.ranking.toUpperCase()}
        </div>
      )}

      <div className="results-list">
        {response?.results.map((r) => (
          <ResultCard key={r.document_id} result={r} />
        ))}
        {response && response.results.length === 0 && (
          <div className="empty-state">No results found. Try a different query.</div>
        )}
      </div>

      {response && totalPages > 1 && (
        <div className="pagination">
          <button disabled={response.page <= 1} onClick={() => runSearch(response.page - 1)}>
            Previous
          </button>
          <span>
            Page {response.page} of {totalPages}
          </span>
          <button disabled={response.page >= totalPages} onClick={() => runSearch(response.page + 1)}>
            Next
          </button>
        </div>
      )}
    </div>
  );
}
