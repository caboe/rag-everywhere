# backend/routers/documents.py
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from typing import List, Optional
import logging
from pathlib import Path
import pypdf
import docx # python-docx

# Assuming main.py defines db, chroma_client, embedding_model
# We need to import them or use dependency injection
from main import db, chroma_client, embedding_model, logger # Direct import from main
from models.mongo_models import DocumentMetadata # Direct import from models

router = APIRouter()

UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "/app/uploads"))

# Ensure upload folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# --- Helper Functions (Placeholder - will be expanded) ---

async def process_document_chunks(doc_metadata: DocumentMetadata):
    """Extracts text, chunks, creates embeddings, and stores them in ChromaDB."""
    logger.info(f"Processing document {doc_metadata.filename} (ID: {doc_metadata.id})")
    extracted_text = ""
    file_path = Path(doc_metadata.filepath)

    if not file_path.exists():
        logger.error(f"File not found at path: {file_path} for document ID: {doc_metadata.id}")
        # TODO: Update status in MongoDB?
        return

    try:
        # 1. Extract text based on mimetype
        logger.info(f"Extracting text from {doc_metadata.filename} (MIME: {doc_metadata.mimetype})")
        if doc_metadata.mimetype == "application/pdf":
            reader = pypdf.PdfReader(str(file_path))
            text_parts = [page.extract_text() for page in reader.pages if page.extract_text()]
            extracted_text = "\n\n".join(text_parts) # Join pages with double newline
        elif doc_metadata.mimetype == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(str(file_path))
            text_parts = [para.text for para in doc.paragraphs if para.text]
            extracted_text = "\n\n".join(text_parts)
        elif doc_metadata.mimetype and doc_metadata.mimetype.startswith("text/"):
             # Handle plain text files
             extracted_text = file_path.read_text(encoding='utf-8') # Specify encoding
        else:
            logger.warning(f"Unsupported file type: {doc_metadata.mimetype} for file {doc_metadata.filename}")
            # TODO: Update status in MongoDB?
            return # Skip processing for unsupported types

        if not extracted_text.strip():
            logger.warning(f"No text extracted from {doc_metadata.filename}")
            # TODO: Update status in MongoDB?
            return

        logger.info(f"Successfully extracted text from {doc_metadata.filename} (Length: {len(extracted_text)})")

        # 2. Chunk text (Placeholder - using simple splitting for now)
        # TODO: Implement more sophisticated chunking (e.g., RecursiveCharacterTextSplitter)
        chunk_size = 1000 # Example size
        overlap = 100    # Example overlap
        chunks = [extracted_text[i:i + chunk_size] for i in range(0, len(extracted_text), chunk_size - overlap)]
        logger.info(f"Split text into {len(chunks)} chunks.")

        if not chunks:
            logger.warning(f"Text splitting resulted in zero chunks for {doc_metadata.filename}")
            return

        # 3. Generate embeddings
        if not embedding_model:
             logger.error("Embedding model not loaded. Cannot generate embeddings.")
             # TODO: Update status in MongoDB?
             return

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, show_progress_bar=False) # Disable progress bar for logs
        logger.info(f"Generated {len(embeddings)} embeddings.")

        # 4. Store in ChromaDB
        if not chroma_client:
            logger.error("ChromaDB client not available. Cannot store embeddings.")
            # TODO: Update status in MongoDB?
            return

        try:
            collection = chroma_client.get_or_create_collection("rag_vectors") # Use a consistent collection name
            doc_id_str = str(doc_metadata.id) # Ensure ID is string for Chroma metadata/ID

            ids = [f"{doc_id_str}_{i}" for i in range(len(chunks))]
            metadatas = [{"document_id": doc_id_str, "chunk_index": i, "filename": doc_metadata.filename} for i in range(len(chunks))]

            logger.info(f"Adding {len(chunks)} vectors to ChromaDB collection 'rag_vectors'...")
            collection.add(
                embeddings=embeddings.tolist(), # Convert numpy array if necessary
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            logger.info(f"Successfully added vectors for {doc_metadata.filename} to ChromaDB.")

        except Exception as e_chroma:
            logger.error(f"Failed to add vectors to ChromaDB for {doc_metadata.filename}: {e_chroma}", exc_info=True)
            # TODO: Update status in MongoDB?
            return

    except Exception as e_process:
        logger.error(f"Error processing document {doc_metadata.filename} (ID: {doc_metadata.id}): {e_process}", exc_info=True)
        # TODO: Update status in MongoDB?
        return

    logger.info(f"Finished processing document {doc_metadata.filename} (ID: {doc_metadata.id})")
    # TODO: Update status in MongoDB to 'processed'

# --- API Endpoints ---

@router.post("/upload", status_code=201)
async def upload_documents(
    files: List[UploadFile] = File(...),
    parent_id: Optional[str] = Form(None) # Get parent_id from form data
    # Add dependency injection for DB etc. if not using globals from main
    # db: AsyncIOMotorDatabase = Depends(get_database)
):
    """
    Handles uploading one or more files.
    Stores the original file, creates metadata in MongoDB,
    and triggers background processing for chunking/embedding.
    """
    if not db:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    uploaded_doc_ids = []
    for file in files:
        try:
            # Sanitize filename (optional, but recommended)
            safe_filename = Path(file.filename).name # Basic sanitization
            if not safe_filename:
                logger.warning(f"Skipping file with invalid name: {file.filename}")
                continue

            # Define save path
            save_path = UPLOAD_FOLDER / safe_filename
            logger.info(f"Receiving file: {safe_filename}, saving to: {save_path}")

            # Save the file to the persistent volume
            try:
                with save_path.open("wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
            finally:
                file.file.close() # Ensure file handle is closed

            # Create metadata entry in MongoDB
            doc_metadata = DocumentMetadata(
                filename=safe_filename,
                filepath=str(save_path), # Store path relative to container root or volume mount
                mimetype=file.content_type,
                parent_id=parent_id,
                is_folder=False # Uploaded files are not folders
            )
            insert_result = await db["documents"].insert_one(doc_metadata.model_dump(by_alias=True))
            doc_id = str(insert_result.inserted_id)
            uploaded_doc_ids.append(doc_id)
            logger.info(f"Successfully saved file {safe_filename} and created metadata (ID: {doc_id})")

            # TODO: Trigger background task for processing (chunking, embedding)
            # For now, call it directly (will block the request)
            # In production, use BackgroundTasks or a task queue (Celery, RQ)
            doc_metadata.id = insert_result.inserted_id # Add the ID back for processing func
            await process_document_chunks(doc_metadata)

        except Exception as e:
            logger.error(f"Failed to process file {file.filename}: {e}", exc_info=True)
            # Optional: Clean up saved file if metadata creation failed?
            # if 'save_path' in locals() and save_path.exists():
            #     save_path.unlink()
            # Continue to next file or raise exception? For now, continue.
            continue # Skip to the next file on error

    if not uploaded_doc_ids:
         raise HTTPException(status_code=400, detail="No files were successfully processed.")

    return {"message": f"Successfully uploaded {len(uploaded_doc_ids)} file(s).", "document_ids": uploaded_doc_ids}

# TODO: Add endpoints for /documents/tree, /documents/select etc.