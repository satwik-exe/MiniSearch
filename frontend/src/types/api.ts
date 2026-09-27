export interface DocumentOut {
  id: number;
  filename: string;
  title: string;
  file_type: string;
  upload_ts: string;
  size_bytes: number;
  word_count: number;
  token_count: number;
}

export interface DocumentDetail extends DocumentOut {
  text: string;
}

export interface UploadResponse {
  document: DocumentOut;
  message: string;
}

export type RankingStrategy = 'tfidf' | 'bm25';

export interface SearchRequest {
  query: string;
  ranking: RankingStrategy;
  top_k: number;
  page: number;
}

export interface SearchResultItem {
  document_id: number;
  title: string;
  filename: string;
  score: number;
  snippet: string;
  matched_terms: string[];
}

export interface SearchResponse {
  query: string;
  ranking: RankingStrategy;
  results: SearchResultItem[];
  total_matches: number;
  page: number;
  top_k: number;
  search_time_ms: number;
}

export interface StatsResponse {
  documents_indexed: number;
  unique_terms: number;
  total_tokens: number;
  average_doc_length: number;
  index_size_bytes: number;
  last_rebuild_ts: string | null;
  last_index_build_ms: number | null;
  last_search_latency_ms: number | null;
}

export interface RebuildResponse {
  documents_indexed: number;
  unique_terms: number;
  total_tokens: number;
  build_time_ms: number;
}
