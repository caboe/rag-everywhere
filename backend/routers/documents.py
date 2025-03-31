# backend/routers/documents.py
import os
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form, BackgroundTasks
from typing import List, Optional
import logging
from pathlib import Path
import pypdf
import docx # python-docx
from pydantic import BaseModel
from langchain_text_splitters import RecursiveCharacterTextSplitter # Import the splitter
import motor.motor_asyncio # Add motor import for type hint
import chromadb # Add chromadb import for type hint
from sentence_transformers import SentenceTransformer # Add SentenceTransformer import for type hint

# Assuming main.py defines db, chroma_client, embedding_model
# We need to import them or use dependency injection
# Import only logger and dependency functions from main
from main import logger, get_db, get_embedding_model, get_chroma_client
from models.mongo_models import DocumentMetadata # Direct import from models

router = APIRouter()

UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", "/app/uploads"))

# Ensure upload folder exists
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)

# --- Helper Functions (Placeholder - will be expanded) ---

async def process_document_chunks(
    doc_metadata: DocumentMetadata,
    embedding_model: SentenceTransformer, # Pass embedding model explicitly
    chroma_client: chromadb.HttpClient # Pass chroma client explicitly
):
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

        # 2. Chunk text using RecursiveCharacterTextSplitter
        chunk_size = int(os.getenv("CHUNK_SIZE", "1000"))
        chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "150"))

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            is_separator_regex=False,
            separators=["\n\n", "\n", " ", ""] # Common separators
        )

        chunks = text_splitter.split_text(extracted_text)
        logger.info(f"Split text into {len(chunks)} chunks using RecursiveCharacterTextSplitter.")

        if not chunks:
            logger.warning(f"Text splitting resulted in zero chunks for {doc_metadata.filename}")
            return

        # 3. Generate embeddings
        # No need to check embedding_model here, it's guaranteed by the caller (upload_documents)

        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, show_progress_bar=False) # Disable progress bar for logs
        logger.info(f"Generated {len(embeddings)} embeddings.")

        # 4. Store in ChromaDB
        # No need to check chroma_client here, it's guaranteed by the caller (upload_documents)

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

@router.post("/upload", status_code=202) # Changed to 202 Accepted for background task
async def upload_documents(
    background_tasks: BackgroundTasks, # Inject BackgroundTasks
    files: List[UploadFile] = File(...),
    parent_id: Optional[str] = Form(None), # Get parent_id from form data
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db), # Inject DB
    embedding_model: SentenceTransformer = Depends(get_embedding_model), # Inject Embedding Model
    chroma_client: chromadb.HttpClient = Depends(get_chroma_client) # Inject Chroma Client
):
    """
    Handles uploading one or more files.
    Stores the original file, creates metadata in MongoDB,
    and triggers background processing for chunking/embedding.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")
    # Also check injected dependencies needed for background task
    if embedding_model is None:
        raise HTTPException(status_code=503, detail="Embedding model not available.")
    if chroma_client is None:
        raise HTTPException(status_code=503, detail="ChromaDB client not available.")


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

            # Add processing task to background
            doc_metadata.id = insert_result.inserted_id # Add the ID back for processing func
            logger.info(f"Adding background task for processing document ID: {doc_id}")
            # Pass the injected dependencies to the background task
            background_tasks.add_task(
                process_document_chunks,
                doc_metadata=doc_metadata,
                embedding_model=embedding_model,
                chroma_client=chroma_client
            )
            # Note: process_document_chunks is now an async function added to the background,
            # FastAPI handles awaiting it after the response is sent.

        except Exception as e:
            logger.error(f"Failed to save file or create metadata for {file.filename}: {e}", exc_info=True)
            # Optional: Clean up saved file if metadata creation failed?
            # if 'save_path' in locals() and save_path.exists():
            #     save_path.unlink()
            # Continue to next file or raise exception? For now, continue.
            continue # Skip to the next file on error

    if not uploaded_doc_ids:
         raise HTTPException(status_code=400, detail="No files were successfully processed.")

    # Return 202 Accepted status code as processing happens in the background
    return {"message": f"Upload accepted for {len(uploaded_doc_ids)} file(s). Processing started.", "document_ids": uploaded_doc_ids}


@router.get("/tree", response_model=List[DocumentMetadata])
async def get_document_tree(db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db)): # Inject DB
    """
    Retrieves the entire document/folder structure from MongoDB.
    The frontend will be responsible for constructing the tree view from this flat list.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    try:
        documents_cursor = db["documents"].find({})
        # Use DocumentMetadata for validation and serialization
        documents = [DocumentMetadata(**doc) async for doc in documents_cursor]
        logger.info(f"Retrieved {len(documents)} items for the document tree.")
        return documents
    except Exception as e:
        logger.error(f"Failed to retrieve document tree from MongoDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document tree.")


class SelectInput(BaseModel):
    document_id: str
    selected: bool
    # Optional: Add recursive flag later if needed for folders
    # recursive: bool = False

@router.put("/select", status_code=200)
async def select_document_for_rag(select_input: SelectInput, db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db)): # Inject DB
    """
    Updates the RAG selection status for a specific document/folder.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    doc_id = select_input.document_id
    selected_status = select_input.selected

    try:
        # Convert string ID back to ObjectId for MongoDB query
        from bson import ObjectId
        try:
            object_id = ObjectId(doc_id)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid document_id format: {doc_id}")

        # TODO: Implement recursive update later if needed for folders
        # if select_input.recursive: ...

        logger.info(f"Updating RAG selection status for document {doc_id} to {selected_status}")
        update_result = await db["documents"].update_one(
            {"_id": object_id},
            {"$set": {"selected_for_rag": selected_status}}
        )

        if update_result.matched_count == 0:
            raise HTTPException(status_code=404, detail=f"Document not found with ID: {doc_id}")

        if update_result.modified_count == 0 and update_result.matched_count == 1:
             # This means the status was already set to the desired value
             logger.info(f"Document {doc_id} RAG selection status was already {selected_status}.")
             # Return 200 OK, but indicate no change occurred if needed by frontend
             return {"message": f"Document {doc_id} status already set to {selected_status}."}

        logger.info(f"Successfully updated RAG selection status for document {doc_id}.")
        return {"message": f"Successfully updated document {doc_id} selection status to {selected_status}."}

    except HTTPException as http_exc:
         raise http_exc # Re-raise specific HTTP exceptions
    except Exception as e:
        logger.error(f"Failed to update document selection status for {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update document selection status.")


# TODO: Add more endpoints if needed