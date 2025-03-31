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
    db = get_db() # Get DB instance - Note: This assumes get_db() is synchronous or handled correctly in background
    if not db:
         logger.error(f"Could not get DB instance to update status for {doc_metadata.filename}")
         # Cannot proceed without DB to update status
         return

    # --- Update status to 'processing' ---
    try:
        await db["documents"].update_one(
            {"_id": doc_metadata.id},
            {"$set": {"processing_status": "processing", "updated_at": datetime.utcnow()}}
        )
        logger.info(f"Set status to 'processing' for {doc_metadata.filename}")
    except Exception as e_status:
        logger.error(f"Failed to update status to 'processing' for {doc_metadata.filename}: {e_status}")
        # Continue processing, but log the error

    extracted_text = ""
    file_path = Path(doc_metadata.filepath)
    status_to_set = "failed" # Default to failed unless explicitly completed or unsupported

    if not file_path.exists():
        logger.error(f"File not found at path: {file_path} for document ID: {doc_metadata.id}")
        status_to_set = "failed"
        # Update status before returning
        try:
            await db["documents"].update_one({"_id": doc_metadata.id}, {"$set": {"processing_status": status_to_set, "updated_at": datetime.utcnow()}})
        except Exception as e_final_status:
            logger.error(f"Failed to set final status '{status_to_set}' for {doc_metadata.filename}: {e_final_status}")
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
            status_to_set = "unsupported"
            # Update status before returning
            try:
                await db["documents"].update_one({"_id": doc_metadata.id}, {"$set": {"processing_status": status_to_set, "updated_at": datetime.utcnow()}})
            except Exception as e_final_status:
                 logger.error(f"Failed to set final status '{status_to_set}' for {doc_metadata.filename}: {e_final_status}")
            return # Skip processing for unsupported types

        if not extracted_text.strip():
            logger.warning(f"No text extracted from {doc_metadata.filename}")
            status_to_set = "failed" # Or maybe "empty"? Let's use "failed" for now.
             # Update status before returning
            try:
                await db["documents"].update_one({"_id": doc_metadata.id}, {"$set": {"processing_status": status_to_set, "updated_at": datetime.utcnow()}})
            except Exception as e_final_status:
                 logger.error(f"Failed to set final status '{status_to_set}' for {doc_metadata.filename}: {e_final_status}")
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
            status_to_set = "completed" # Mark as completed if vectors added successfully

        except Exception as e_chroma:
            logger.error(f"Failed to add vectors to ChromaDB for {doc_metadata.filename}: {e_chroma}", exc_info=True)
            status_to_set = "failed" # Set status to failed on Chroma error
            # Do not return here, let the final status update happen

    except Exception as e_process:
        logger.error(f"Error processing document {doc_metadata.filename} (ID: {doc_metadata.id}): {e_process}", exc_info=True)
        status_to_set = "failed" # Set status to failed on general processing error
        # Do not return here, let the final status update happen

    # --- Final Status Update ---
    try:
        await db["documents"].update_one(
            {"_id": doc_metadata.id},
            {"$set": {"processing_status": status_to_set, "updated_at": datetime.utcnow()}}
        )
        logger.info(f"Set final status to '{status_to_set}' for {doc_metadata.filename} (ID: {doc_metadata.id})")
    except Exception as e_final_status:
        logger.error(f"Failed to set final status '{status_to_set}' for {doc_metadata.filename}: {e_final_status}")

    logger.info(f"Finished processing attempt for document {doc_metadata.filename} (ID: {doc_metadata.id}) with status: {status_to_set}")

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
    recursive: bool = False # Add recursive flag

@router.put("/select", status_code=200)
async def select_document_for_rag(select_input: SelectInput, db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db)): # Inject DB
    """
    Updates the RAG selection status for a specific document/folder.
    If 'recursive' is true and the target is a folder, updates recursively.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    doc_id = select_input.document_id
    selected_status = select_input.selected
    recursive = select_input.recursive # Get the recursive flag

    try:
        # Convert string ID back to ObjectId for MongoDB query
        from bson import ObjectId
        try:
            object_id = ObjectId(doc_id)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid document_id format: {doc_id}")

        # Check if the target item exists
        target_item = await db["documents"].find_one({"_id": object_id})
        if not target_item:
            raise HTTPException(status_code=404, detail=f"Document or folder not found with ID: {doc_id}")

        ids_to_update = [object_id] # Start with the target item itself
        modified_count = 0 # Initialize modified count

        # If recursive update is requested for a folder
        if recursive and target_item.get("is_folder"):
            logger.info(f"Performing recursive RAG selection update for folder {doc_id} to {selected_status}")
            # Find all descendant IDs (including subfolders and files) using an async generator
            async def find_descendant_ids(parent_id_str):
                children_cursor = db["documents"].find({"parent_id": parent_id_str})
                async for child in children_cursor:
                    child_id_str = str(child["_id"])
                    try:
                        yield ObjectId(child_id_str) # Yield ObjectId
                        if child.get("is_folder"):
                            # Recursively yield from subfolders
                            async for sub_child_id in find_descendant_ids(child_id_str):
                                yield sub_child_id
                    except Exception:
                         logger.warning(f"Invalid child ObjectId format encountered during recursive select: {child_id_str}")
                         continue

            async for desc_id in find_descendant_ids(doc_id):
                ids_to_update.append(desc_id)

            logger.info(f"Found {len(ids_to_update)} items (including descendants) to update for folder {doc_id}.")

            # Perform bulk update
            if ids_to_update: # Ensure there are IDs to update
                update_result = await db["documents"].update_many(
                    {"_id": {"$in": ids_to_update}},
                    {"$set": {"selected_for_rag": selected_status}}
                )
                modified_count = update_result.modified_count
                message = f"Successfully updated RAG selection status for folder {doc_id} and {modified_count} descendants to {selected_status}."
            else:
                 # Should not happen if the initial folder exists, but handle defensively
                 message = f"No items found to update for folder {doc_id}."


        else:
            # Single item update (or recursive=False, or target is a file)
            if recursive and not target_item.get("is_folder"):
                 logger.warning(f"Recursive selection requested for a file (ID: {doc_id}). Performing non-recursive update.")

            logger.info(f"Updating RAG selection status for single item {doc_id} to {selected_status}")
            update_result = await db["documents"].update_one(
                {"_id": object_id},
                {"$set": {"selected_for_rag": selected_status}}
            )
            modified_count = update_result.modified_count

            if update_result.matched_count == 0:
                 # This case should be caught by the initial find_one check, but added for safety
                 raise HTTPException(status_code=404, detail=f"Document not found with ID: {doc_id}")

            if modified_count == 0 and update_result.matched_count == 1:
                 message = f"Document {doc_id} RAG selection status was already {selected_status}."
            else:
                 message = f"Successfully updated document {doc_id} selection status to {selected_status}."


        logger.info(message)
        # Return modified_count in the response body
        return {"message": message, "modified_count": modified_count}

    except HTTPException as http_exc:
         raise http_exc # Re-raise specific HTTP exceptions
    except Exception as e:
        logger.error(f"Failed to update document selection status for {doc_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to update document selection status.")


# --- Pydantic Model for Folder Creation ---
class CreateFolderInput(BaseModel):
    folder_name: str
    parent_id: Optional[str] = None # ID of the parent folder, None for root

@router.post("/folder", response_model=DocumentMetadata, status_code=201)
async def create_folder(
    folder_input: CreateFolderInput,
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db) # Inject DB
):
    """
    Creates a new folder in the document structure.
    """
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    folder_name = folder_input.folder_name.strip()
    if not folder_name:
        raise HTTPException(status_code=400, detail="Folder name cannot be empty.")

    # Optional: Check if a folder/file with the same name already exists under the same parent
    # query = {"filename": folder_name, "parent_id": folder_input.parent_id, "is_folder": True}
    # existing = await db["documents"].find_one(query)
    # if existing:
    #     raise HTTPException(status_code=409, detail=f"A folder named '{folder_name}' already exists in this location.")

    try:
        folder_metadata = DocumentMetadata(
            filename=folder_name,
            filepath=None, # Folders don't have a direct file path
            mimetype=None, # Folders don't have a mimetype
            parent_id=folder_input.parent_id,
            is_folder=True,
            selected_for_rag=False # Default selection status for new folders
            # upload_date is handled by default_factory
        )
        insert_result = await db["documents"].insert_one(folder_metadata.model_dump(by_alias=True))
        created_folder = await db["documents"].find_one({"_id": insert_result.inserted_id})

        if not created_folder:
             # This should ideally not happen if insert succeeded
             logger.error(f"Failed to retrieve created folder immediately after insertion for {folder_name}")
             raise HTTPException(status_code=500, detail="Failed to create folder.")

        logger.info(f"Successfully created folder '{folder_name}' (ID: {insert_result.inserted_id}) under parent '{folder_input.parent_id}'.")
        # Use DocumentMetadata to validate and return the created folder data
        return DocumentMetadata(**created_folder)

    except Exception as e:
        logger.error(f"Failed to create folder '{folder_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to create folder.")


# --- Helper Function for Recursive Deletion ---

async def delete_item_recursive(
    item_id: str,
    db: motor.motor_asyncio.AsyncIOMotorDatabase,
    chroma_client: chromadb.HttpClient,
    upload_folder: Path # Pass UPLOAD_FOLDER path
):
    """
    Recursively deletes an item (file or folder) and its descendants.
    Deletes metadata from MongoDB, vectors from ChromaDB, and original files.
    """
    logger.info(f"Attempting recursive delete starting from item ID: {item_id}")
    items_to_delete_mongo_ids = []
    files_to_delete_paths = []
    doc_ids_to_delete_vectors = [] # Store string IDs for Chroma query

    # Use a queue for traversal (e.g., breadth-first)
    queue = [item_id]
    processed_ids = set() # To handle potential cycles

    while queue:
        current_id_str = queue.pop(0)
        if current_id_str in processed_ids:
            continue
        processed_ids.add(current_id_str)

        try:
            from bson import ObjectId
            current_object_id = ObjectId(current_id_str)
        except Exception:
            logger.warning(f"Invalid ObjectId format encountered during delete: {current_id_str}")
            continue # Skip invalid IDs

        item = await db["documents"].find_one({"_id": current_object_id})
        if not item:
            logger.warning(f"Item not found during recursive delete: {current_id_str}")
            continue # Item might have been deleted already

        items_to_delete_mongo_ids.append(current_object_id) # Add the MongoDB ObjectId for deletion

        if item.get("is_folder"):
            logger.debug(f"Item {current_id_str} is a folder. Finding children...")
            # Find direct children and add their string IDs to the queue
            children_cursor = db["documents"].find({"parent_id": current_id_str})
            async for child in children_cursor:
                child_id_str = str(child["_id"])
                if child_id_str not in processed_ids:
                    queue.append(child_id_str)
        else:
            # It's a file, mark its vectors (using string ID) and file path for deletion
            logger.debug(f"Item {current_id_str} is a file. Marking for vector and file deletion.")
            doc_ids_to_delete_vectors.append(current_id_str) # Use string ID for Chroma metadata
            if item.get("filepath"):
                 # Ensure the path is absolute based on the UPLOAD_FOLDER base
                 file_path_to_delete = Path(item["filepath"])
                 # Make sure it's relative to the upload folder base if stored relatively,
                 # or handle absolute paths correctly. Assuming absolute paths stored for now.
                 if file_path_to_delete.is_absolute() and file_path_to_delete.exists():
                      files_to_delete_paths.append(file_path_to_delete)
                 elif not file_path_to_delete.is_absolute():
                      # This case assumes filepath is stored relative to UPLOAD_FOLDER
                      absolute_path = upload_folder / file_path_to_delete
                      if absolute_path.exists():
                           files_to_delete_paths.append(absolute_path)
                      else:
                           logger.warning(f"Constructed absolute path {absolute_path} does not exist for relative path {file_path_to_delete}")
                 else: # Absolute path stored, but does not exist
                      logger.warning(f"File path {item['filepath']} for item {current_id_str} does not exist.")


    # --- Perform Deletions ---

    # 1. Delete vectors from ChromaDB
    if doc_ids_to_delete_vectors:
        try:
            collection_name = os.getenv("CHROMA_COLLECTION", "rag_vectors")
            # Need to handle potential errors if collection doesn't exist
            try:
                collection = chroma_client.get_collection(collection_name)
                logger.info(f"Deleting vectors from ChromaDB for {len(doc_ids_to_delete_vectors)} document IDs: {doc_ids_to_delete_vectors}")
                # Delete by filtering on the document_id metadata field which stores the string representation of the Mongo ObjectId
                collection.delete(where={"document_id": {"$in": doc_ids_to_delete_vectors}})
                logger.info(f"Successfully deleted vectors for {len(doc_ids_to_delete_vectors)} documents.")
            except Exception as e_get_coll: # Catch specific Chroma exceptions if possible
                 logger.error(f"Failed to get Chroma collection '{collection_name}' during delete: {e_get_coll}", exc_info=True)
                 # Continue with other deletions even if vector deletion fails

        except Exception as e_chroma_del:
            logger.error(f"Failed to delete vectors from ChromaDB: {e_chroma_del}", exc_info=True)
            # Continue with other deletions

    # 2. Delete original files from the filesystem
    if files_to_delete_paths:
        logger.info(f"Deleting {len(files_to_delete_paths)} original files from filesystem...")
        for file_path in files_to_delete_paths:
            try:
                if file_path.is_file(): # Check if it's actually a file before deleting
                    file_path.unlink()
                    logger.debug(f"Deleted file: {file_path}")
                elif file_path.exists(): # Check if it exists but isn't a file (e.g., a directory mistakenly referenced)
                    logger.warning(f"Path marked for deletion exists but is not a file: {file_path}")
                # No need for an else here, if it doesn't exist, we already warned during path collection
            except OSError as e_file_del:
                logger.error(f"Failed to delete file {file_path}: {e_file_del}", exc_info=True)
                # Continue deleting other files even if one fails

    # 3. Delete metadata from MongoDB
    if items_to_delete_mongo_ids:
        try:
            logger.info(f"Deleting {len(items_to_delete_mongo_ids)} items from MongoDB...")
            delete_result = await db["documents"].delete_many({"_id": {"$in": items_to_delete_mongo_ids}})
            logger.info(f"MongoDB delete result: {delete_result.deleted_count} items deleted.")
        except Exception as e_mongo_del:
            logger.error(f"Failed to delete items from MongoDB: {e_mongo_del}", exc_info=True)
            # This is a critical failure, but the process is already underway.

    logger.info(f"Finished recursive delete process starting from item ID: {item_id}")


@router.delete("/{document_id}", status_code=202) # Use 202 Accepted for background task
async def delete_document_or_folder(
    document_id: str,
    background_tasks: BackgroundTasks, # Inject BackgroundTasks
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db), # Inject DB
    chroma_client: chromadb.HttpClient = Depends(get_chroma_client) # Inject Chroma Client
):
    """
    Deletes a document or folder (recursively) by its ID.
    Triggers a background task to handle the deletion process.
    """
    # Basic validation of dependencies
    if db is None:
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")
    if chroma_client is None:
        raise HTTPException(status_code=503, detail="ChromaDB client not available.")

    # Validate ObjectId format before starting background task
    from bson import ObjectId
    try:
        object_id_to_delete = ObjectId(document_id)
    except Exception:
        raise HTTPException(status_code=400, detail=f"Invalid document_id format: {document_id}")

    # Check if the item exists before queueing deletion
    item_exists = await db["documents"].find_one({"_id": object_id_to_delete})
    if not item_exists:
         raise HTTPException(status_code=404, detail=f"Document or folder not found with ID: {document_id}")

    logger.info(f"Queueing background task for recursive deletion of item ID: {document_id}")
    background_tasks.add_task(
        delete_item_recursive,
        item_id=document_id, # Pass the string ID to the background task
        db=db,
        chroma_client=chroma_client,
        upload_folder=UPLOAD_FOLDER # Pass the UPLOAD_FOLDER path constant
    )

    return {"message": f"Deletion process for item {document_id} started in background."}


# TODO: Add more endpoints if needed