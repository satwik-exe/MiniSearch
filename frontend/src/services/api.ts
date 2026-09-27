import type {
  DocumentDetail,
  DocumentOut,
  RebuildResponse,
  SearchRequest,
  SearchResponse,
  StatsResponse,
  UploadResponse,
} from '../types/api';

const BASE = '/api';

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
    } catch {
      // response body wasn't JSON; fall back to statusText
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

export async function uploadDocument(file: File): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);
  const res = await fetch(`${BASE}/documents/upload`, { method: 'POST', body: formData });
  return handle<UploadResponse>(res);
}

export async function listDocuments(): Promise<DocumentOut[]> {
  const res = await fetch(`${BASE}/documents`);
  return handle<DocumentOut[]>(res);
}

export async function getDocument(id: number): Promise<DocumentDetail> {
  const res = await fetch(`${BASE}/documents/${id}`);
  return handle<DocumentDetail>(res);
}

export async function deleteDocument(id: number): Promise<void> {
  const res = await fetch(`${BASE}/documents/${id}`, { method: 'DELETE' });
  await handle<{ message: string }>(res);
}

export async function search(req: SearchRequest): Promise<SearchResponse> {
  const res = await fetch(`${BASE}/search`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(req),
  });
  return handle<SearchResponse>(res);
}

export async function getStats(): Promise<StatsResponse> {
  const res = await fetch(`${BASE}/stats`);
  return handle<StatsResponse>(res);
}

export async function rebuildIndex(): Promise<RebuildResponse> {
  const res = await fetch(`${BASE}/index/rebuild`, { method: 'POST' });
  return handle<RebuildResponse>(res);
}
