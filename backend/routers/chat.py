# backend/routers/chat.py
import os
import logging
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel
from typing import List, Optional
import motor.motor_asyncio # Add motor import for type hint
from sentence_transformers import SentenceTransformer # Add for type hint
import chromadb # Add for type hint
import ollama # Add for type hint

# Assuming main.py initializes these and they are accessible
# In a real app, you might use dependency injection more formally
# Remove db, keep others for now (will refactor later if needed)
from main import logger, get_db, get_embedding_model, get_chroma_client, get_ollama_client # Import get_ollama_client, remove ollama_client

router = APIRouter()

# --- Pydantic Models ---

# Model for the internal /api/chat endpoint
class InternalChatMessageInput(BaseModel):
    message: str

class InternalChatMessageOutput(BaseModel):
    response: str
    sources: Optional[List[dict]] = None

# Models for the OpenAI-compatible /v1/chat/completions endpoint
class OpenAIMessage(BaseModel):
    role: str # Typically "user", "assistant", "system"
    content: str

class OpenAIChatCompletionInput(BaseModel):
    model: str # Model name (can be ignored or validated against env var)
    messages: List[OpenAIMessage]
    stream: Optional[bool] = False # Streaming not implemented yet
    # Add other common OpenAI params if needed (temperature, max_tokens, etc.)

class OpenAIChoice(BaseModel):
    index: int = 0
    message: OpenAIMessage
    finish_reason: str = "stop" # Assuming simple stop for now

class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0 # Placeholder
    completion_tokens: int = 0 # Placeholder
    total_tokens: int = 0 # Placeholder

class OpenAIChatCompletionOutput(BaseModel):
    id: str # Generate a unique ID
    object: str = "chat.completion"
    created: int # Unix timestamp
    model: str # Model name used
    choices: List[OpenAIChoice]
    usage: OpenAIUsage

# --- Helper Functions (Placeholder - RAG logic goes here) ---

async def perform_rag(
    query: str,
    db: motor.motor_asyncio.AsyncIOMotorDatabase, # Accept resolved DB
    embedding_model: SentenceTransformer,         # Accept resolved model
    chroma_client: chromadb.HttpClient,           # Accept resolved Chroma client
    ollama_client: ollama.AsyncClient             # Accept resolved Ollama client
) -> InternalChatMessageOutput:
    """
    Placeholder for the actual RAG logic.
    1. Get selected document IDs from MongoDB.
    2. Embed the user query.
    3. Query ChromaDB with filtering.
    4. Format prompt with context.
    5. Call Ollama LLM.
    6. Format and return response.
    """
    logger.info(f"Received chat query: '{query}'")

    query_embedding = None # Initialize query_embedding before try block

    # --- Implement RAG Steps ---
    # 1. Get selected docs from MongoDB, considering tree structure
    final_selected_doc_ids = set() # Use a set to avoid duplicates
    if db is None: # Corrected check
         raise HTTPException(status_code=503, detail="MongoDB client not connected")
    try:
        logger.info("Fetching all documents/folders to determine RAG selection...")
        all_docs_cursor = db.documents.find({})
        all_items = await all_docs_cursor.to_list(length=None) # Fetch all items into memory
        logger.info(f"Fetched {len(all_items)} total items from DB.")

        # Build a lookup for faster access and parent->children mapping
        items_by_id = {str(item["_id"]): item for item in all_items}
        children_by_parent_id = {}
        for item in all_items:
            parent_id = item.get("parent_id")
            if parent_id:
                 if parent_id not in children_by_parent_id:
                     children_by_parent_id[parent_id] = []
                 children_by_parent_id[parent_id].append(str(item["_id"]))

        # Find initially selected nodes
        directly_selected_ids = {str(item["_id"]) for item in all_items if item.get("selected_for_rag")}
        logger.debug(f"Directly selected node IDs: {directly_selected_ids}")

        # Recursive function to find all descendant files of a folder
        def get_descendant_files(folder_id: str, visited: set):
            descendants = set()
            if folder_id in visited: # Prevent cycles
                 return descendants
            visited.add(folder_id)

            if folder_id in children_by_parent_id:
                for child_id in children_by_parent_id[folder_id]:
                    if child_id in items_by_id:
                        child_item = items_by_id[child_id]
                        if not child_item.get("is_folder"):
                            descendants.add(child_id)
                        else:
                            # Recursively get descendants of subfolders
                            descendants.update(get_descendant_files(child_id, visited))
            return descendants

        # Iterate through directly selected nodes and collect all relevant file IDs
        for selected_id in directly_selected_ids:
            if selected_id in items_by_id:
                item = items_by_id[selected_id]
                if not item.get("is_folder"):
                    final_selected_doc_ids.add(selected_id) # Add directly selected files
                else:
                    # If it's a folder, find all descendant files
                    descendant_files = get_descendant_files(selected_id, set())
                    final_selected_doc_ids.update(descendant_files)
                    logger.debug(f"Folder {selected_id} selected, adding descendant files: {descendant_files}")

        selected_doc_ids = list(final_selected_doc_ids) # Convert set back to list for Chroma query

        if not selected_doc_ids:
            logger.warning("No documents are effectively selected for RAG after tree traversal.")
            # Return a specific message instead of proceeding with empty context
            # Corrected model name below
            return InternalChatMessageOutput(response="No documents are currently selected for chat. Please select documents or folders in the Upload section.", sources=[])
            # raise HTTPException(status_code=400, detail="No documents selected for chat.") # Alternative

        logger.info(f"Final list of {len(selected_doc_ids)} document IDs selected for RAG (including descendants): {selected_doc_ids}")

    except Exception as e:
        logger.error(f"Failed to determine selected documents from MongoDB tree: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve document selection status")

    # CORRECTED INDENTATION STARTS HERE
    # 2. Embed query (using injected embedding_model)
    logger.debug(f"Attempting to use embedding_model: {embedding_model} (type: {type(embedding_model)})") # Log the model object itself
    try:
        # Use the injected embedding_model directly
        raw_embedding = embedding_model.encode(query)
        logger.debug(f"Raw embedding type: {type(raw_embedding)}, shape/size: {getattr(raw_embedding, 'shape', 'N/A')}") # Log intermediate result
        query_embedding = raw_embedding.tolist()
        logger.debug(f"Query embedding generated (type: {type(query_embedding)}, is None: {query_embedding is None}).") # Log after assignment
    except Exception as e:
        logger.error("!!! Exception occurred during embedding generation !!!") # Log generic message first
        logger.error(f"Failed to generate query embedding: {e}", exc_info=True) # Log exception with traceback
        raise HTTPException(status_code=500, detail="Failed to generate query embedding")

    # Ensure embedding was generated before proceeding
    if query_embedding is None:
        logger.error("Query embedding is None after generation attempt.")
        raise HTTPException(status_code=500, detail="Failed to generate query embedding (result was None)")

    # 3. Query ChromaDB (using injected chroma_client)
    context_chunks = [] # Initialize empty list for context
    # Removed check for chroma_client, as it's handled by dependency injection
    if not selected_doc_ids:
        logger.warning("Skipping ChromaDB query as no documents are selected for RAG.")
    else:
        try:
            collection_name = os.getenv("CHROMA_COLLECTION", "rag_vectors")
            n_results_str = os.getenv("CHROMA_N_RESULTS", "5")
            try:
                n_results = int(n_results_str)
            except ValueError:
                logger.warning(f"Invalid CHROMA_N_RESULTS value '{n_results_str}', defaulting to 5.")
                n_results = 5

            logger.info(f"Querying ChromaDB collection '{collection_name}' for top {n_results} results, filtering by {len(selected_doc_ids)} document IDs.")

            # Ensure collection exists
            try:
                 collection = chroma_client.get_collection(collection_name) # Use correct parameter name
            except Exception as e_coll:
                 logger.error(f"Failed to get ChromaDB collection '{collection_name}': {e_coll}. Make sure it's created during document upload.")
                 # Depending on requirements, could try get_or_create_collection, but it's better if upload creates it.
                 raise HTTPException(status_code=500, detail=f"Vector collection '{collection_name}' not found.")

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                # Filter chunks based on the document IDs retrieved from MongoDB
                where={"document_id": {"$in": selected_doc_ids}}
                # Alternative: Use 'where_document' if filtering by full document content is needed/possible
                # where_document={"$contains": "some_keyword"} # Example
            )

            # Extract the text content of the retrieved chunks
            if results and results.get('documents') and results['documents'][0]:
                 context_chunks = results['documents'][0]
                 logger.info(f"Retrieved {len(context_chunks)} context chunks from ChromaDB.")
                 logger.debug(f"Context chunks: {context_chunks}")
            else:
                 logger.warning("ChromaDB query returned no relevant chunks for the selected documents.")

        except Exception as e:
            logger.error(f"Failed to query ChromaDB: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to query vector database")

    # 4. Format Prompt
    # Removed stray raise HTTPException from previous version
    context_str = "\n".join(context_chunks)
    prompt = f"""Based on the following context, answer the user's question.
Context:
{context_str}

User Question: {query}

Answer:"""
    logger.debug(f"Generated prompt for LLM:\n{prompt}")

    # 5. Call Ollama LLM (using injected ollama_client)
    # Removed check for ollama_client, as it's handled by dependency injection
    try:
        llm_model_name = os.getenv("LLM_MODEL_NAME", "gemma:2b") # Get model from env
        logger.info(f"Sending prompt to Ollama model: {llm_model_name}")
        response = await ollama_client.chat(
            model=llm_model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        llm_response_content = response['message']['content']
        logger.info(f"Received response from Ollama: {llm_response_content[:100]}...") # Log truncated response
    except Exception as e:
        logger.error(f"Failed to get response from Ollama: {e}")
        raise HTTPException(status_code=500, detail="Failed to get response from LLM")

    # 6. Format and return
    show_sources = os.getenv("SHOW_SOURCES", "false").lower() == "true"
    # Corrected model name below
    output = InternalChatMessageOutput(
        response=llm_response_content,
        sources=context_chunks if show_sources else None # Include sources if flag is set
    )
    return output
# CORRECTED INDENTATION ENDS HERE

# --- API Endpoint ---

# Internal API endpoint used by the SvelteKit frontend
@router.post("", response_model=InternalChatMessageOutput) # Removed trailing slash from path
async def handle_internal_chat_message(
    chat_input: InternalChatMessageInput = Body(...),
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db), # Inject DB
    embedding_model: SentenceTransformer = Depends(get_embedding_model), # Inject Model
    chroma_client: chromadb.HttpClient = Depends(get_chroma_client), # Inject Chroma
    ollama_client: ollama.AsyncClient = Depends(get_ollama_client) # Inject Ollama
):
    """
    Handles incoming chat messages, performs RAG, and returns the LLM response.
    """
    try:
        # Call the RAG function, passing the resolved dependencies
        rag_result: InternalChatMessageOutput = await perform_rag(
            query=chat_input.message,
            db=db,
            embedding_model=embedding_model,
            chroma_client=chroma_client,
            ollama_client=ollama_client # Pass the injected client
        )
        return rag_result
    except HTTPException as http_exc:
        # Re-raise HTTPExceptions (like 503 Service Unavailable)
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in internal chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing chat message.")

# OpenAI-compatible endpoint
@router.post("/v1/chat/completions", response_model=OpenAIChatCompletionOutput, tags=["OpenAI"])
async def handle_openai_chat_completion(
    request_body: OpenAIChatCompletionInput = Body(...),
    db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db), # Inject DB
    embedding_model: SentenceTransformer = Depends(get_embedding_model), # Inject Model
    chroma_client: chromadb.HttpClient = Depends(get_chroma_client), # Inject Chroma
    ollama_client: ollama.AsyncClient = Depends(get_ollama_client) # Inject Ollama
):
    """
    Handles chat completion requests compatible with the OpenAI API spec.
    """
    # Extract the last user message as the query
    # TODO: Potentially handle conversation history differently if needed
    user_message = next((msg.content for msg in reversed(request_body.messages) if msg.role == 'user'), None)

    if not user_message:
        raise HTTPException(status_code=400, detail="No user message found in the request.")

    if request_body.stream:
        # TODO: Implement streaming response if needed
        raise HTTPException(status_code=501, detail="Streaming responses are not implemented yet.")

    try:
        # Reuse the core RAG logic, passing the resolved dependencies
        rag_result: InternalChatMessageOutput = await perform_rag(
            query=user_message,
            db=db,
            embedding_model=embedding_model,
            chroma_client=chroma_client,
            ollama_client=ollama_client
        )

        # Format the response according to OpenAI spec
        import time
        import uuid

        llm_model_name = os.getenv("LLM_MODEL_NAME", "gemma:2b") # Get model name again for response

        response_message = OpenAIMessage(role="assistant", content=rag_result.response)
        choice = OpenAIChoice(message=response_message)
        usage = OpenAIUsage() # Using placeholder token counts

        openai_response = OpenAIChatCompletionOutput(
            id=f"chatcmpl-{uuid.uuid4()}",
            created=int(time.time()),
            model=llm_model_name,
            choices=[choice],
            usage=usage
        )
        return openai_response

    except HTTPException as http_exc:
        # Re-raise HTTPExceptions
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred in OpenAI chat endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error processing chat completion request.")