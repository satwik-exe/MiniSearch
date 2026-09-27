from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile

from app.dependencies import get_engine
from app.documents.ingestion import CorruptFile, FileTooLarge, UnsupportedFileType, ingest
from app.models.schemas import DocumentDetail, DocumentOut, UploadResponse
from app.search.engine import SearchEngine

router = APIRouter()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...), engine: SearchEngine = Depends(get_engine)
) -> UploadResponse:
    data = await file.read()
    try:
        ingested = ingest(file.filename or "untitled.txt", data)
    except UnsupportedFileType as e:
        raise HTTPException(status_code=415, detail=str(e))
    except FileTooLarge as e:
        raise HTTPException(status_code=413, detail=str(e))
    except CorruptFile as e:
        # e.g. a corrupted/truncated PDF, or a non-PDF file renamed to .pdf --
        # 422 Unprocessable Entity: the request was well-formed (right field,
        # allowed extension) but its content couldn't actually be processed.
        raise HTTPException(status_code=422, detail=str(e))

    row = engine.add_document(ingested)
    return UploadResponse(
        document=DocumentOut(**row),
        message=f"'{row['title']}' indexed successfully ({row['word_count']} words).",
    )


@router.get("/documents", response_model=list[DocumentOut])
def list_documents(engine: SearchEngine = Depends(get_engine)) -> list[DocumentOut]:
    return [DocumentOut(**d) for d in engine.list_documents()]


@router.get("/documents/{doc_id}", response_model=DocumentDetail)
def get_document(doc_id: int, engine: SearchEngine = Depends(get_engine)) -> DocumentDetail:
    doc = engine.get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentDetail(**doc)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: int, engine: SearchEngine = Depends(get_engine)) -> dict:
    ok = engine.delete_document(doc_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"message": "Document deleted", "document_id": doc_id}
