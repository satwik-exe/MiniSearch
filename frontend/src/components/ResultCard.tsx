import type { SearchResultItem } from '../types/api';

export default function ResultCard({ result }: { result: SearchResultItem }) {
  return (
    <article className="result-card">
      <div className="result-header">
        <h3>{result.title}</h3>
        <span className="score-badge">Score: {result.score.toFixed(3)}</span>
      </div>
      <p className="result-filename">{result.filename}</p>
      <p
        className="result-snippet"
        // Snippet HTML is generated server-side by app/search/snippet.py, which
        // HTML-escapes document content before inserting only its own <mark> tags,
        // so this is safe to render directly.
        dangerouslySetInnerHTML={{ __html: result.snippet }}
      />
      {result.matched_terms.length > 0 && (
        <div className="matched-terms">
          <span className="matched-label">Matched terms:</span>{' '}
          {result.matched_terms.join(' · ')}
        </div>
      )}
    </article>
  );
}
