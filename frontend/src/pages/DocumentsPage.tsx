import { useEffect, useRef, useState, type ChangeEvent } from 'react';
import { deleteDocument, listDocuments, uploadDocument } from '../services/api';
import type { DocumentOut } from '../types/api';

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<DocumentOut[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  async function refresh() {
    try {
      setDocuments(await listDocuments());
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to load documents');
    }
  }

  useEffect(() => {
    refresh();
  }, []);

  async function handleUpload(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setError(null);
    try {
      await uploadDocument(file);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Upload failed');
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  }

  async function handleDelete(id: number) {
    try {
      await deleteDocument(id);
      await refresh();
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Delete failed');
    }
  }

  return (
    <div className="documents-page">
      <div className="upload-panel">
        <label className="upload-button">
          {uploading ? 'Uploading...' : 'Upload Document (.txt, .md, .pdf)'}
          <input
            ref={fileInputRef}
            type="file"
            accept=".txt,.md,.pdf"
            onChange={handleUpload}
            disabled={uploading}
            hidden
          />
        </label>
      </div>

      {error && <div className="error-banner">{error}</div>}

      <table className="documents-table">
        <thead>
          <tr>
            <th>Title</th>
            <th>Type</th>
            <th>Words</th>
            <th>Uploaded</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {documents.map((doc) => (
            <tr key={doc.id}>
              <td>{doc.title}</td>
              <td>
                <span className="type-badge">{doc.file_type}</span>
              </td>
              <td>
                {doc.word_count.toLocaleString()} ({formatBytes(doc.size_bytes)})
              </td>
              <td>{new Date(doc.upload_ts).toLocaleString()}</td>
              <td>
                <button className="delete-button" onClick={() => handleDelete(doc.id)}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
          {documents.length === 0 && (
            <tr>
              <td colSpan={5} className="empty-state">
                No documents uploaded yet.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
