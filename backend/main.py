# backend/main.py
import os
from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv
import logging
import motor.motor_asyncio
import chromadb
from sentence_transformers import SentenceTransformer
import ollama
from contextlib import asynccontextmanager
from typing import Union
from fastapi.middleware.cors import CORSMiddleware # Import CORS middleware

# Load environment variables
load_dotenv()

# Configure logging
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=log_level, # Use level from environment variable
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Global variables for clients and models ---
mongo_client: Union[motor.motor_asyncio.AsyncIOMotorClient, None] = None
db: Union[motor.motor_asyncio.AsyncIOMotorDatabase, None] = None
chroma_client: Union[chromadb.HttpClient, None] = None
embedding_model: Union[SentenceTransformer, None] = None
ollama_client: Union[ollama.AsyncClient, None] = None

# --- Lifespan context manager for startup/shutdown ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global mongo_client, db, chroma_client, embedding_model, ollama_client
    logger.info("Starting up application and initializing resources...")

    # --- MongoDB Connection ---
    mongo_user = os.getenv("MONGO_USER")
    mongo_password = os.getenv("MONGO_PASSWORD")
    mongo_host = os.getenv("MONGO_HOST", "mongo")
    mongo_port = os.getenv("MONGO_PORT", "27017")
    mongo_db_name = os.getenv("MONGO_DB_NAME", "rag_metadata")
    mongo_url = f"mongodb://{mongo_user}:{mongo_password}@{mongo_host}:{mongo_port}"
    try:
        logger.info(f"Connecting to MongoDB at {mongo_host}:{mongo_port}...")
        mongo_client = motor.motor_asyncio.AsyncIOMotorClient(mongo_url)
        # The ismaster command is cheap and does not require auth.
        await mongo_client.admin.command('ismaster')
        db = mongo_client[mongo_db_name]
        logger.info(f"Successfully connected to MongoDB, using database '{mongo_db_name}'.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        mongo_client = None # Ensure client is None if connection failed
        db = None

    # --- ChromaDB Connection ---
    chroma_host = os.getenv("CHROMA_HOST", "chroma")
    chroma_port = os.getenv("CHROMA_PORT", "8000")
    try:
        logger.info(f"Connecting to ChromaDB at {chroma_host}:{chroma_port}...")
        chroma_client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
        # You might add a heartbeat check if the client library supports it easily
        # For now, we assume connection if no exception during init
        # chroma_client.heartbeat() # Example if available
        logger.info("Successfully initialized ChromaDB client.")
        # TODO: Create collection if it doesn't exist?
        # try:
        #     collection = chroma_client.get_or_create_collection("rag_vectors")
        #     logger.info(f"ChromaDB collection 'rag_vectors' ready.")
        # except Exception as e_coll:
        #     logger.error(f"Failed to get/create ChromaDB collection: {e_coll}")

    except Exception as e:
        logger.error(f"Failed to connect to ChromaDB: {e}")
        chroma_client = None

    # --- Load Embedding Model ---
    embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2")
    try:
        logger.info(f"Loading embedding model: {embedding_model_name}...")
        # Specify cache folder if needed, e.g., inside a persistent volume
        # cache_dir = "/app/models_cache"
        # os.makedirs(cache_dir, exist_ok=True)
        embedding_model = SentenceTransformer(embedding_model_name) # add cache_folder=cache_dir if needed
        logger.info("Successfully loaded embedding model.")
    except Exception as e:
        logger.error(f"Failed to load embedding model: {e}")
        embedding_model = None

    # --- Ollama Connection ---
    ollama_api_url = os.getenv("OLLAMA_API_URL", "http://ollama:11434")
    try:
        logger.info(f"Connecting to Ollama at {ollama_api_url}...")
        ollama_client = ollama.AsyncClient(host=ollama_api_url)
        # Perform a basic check, e.g., list models
        await ollama_client.list()
        logger.info("Successfully connected to Ollama.")
        # TODO: Check if the desired LLM_MODEL_NAME exists?
        # llm_model_name = os.getenv("LLM_MODEL_NAME", "gemma:2b")
        # models = await ollama_client.list()
        # if not any(m['name'] == llm_model_name for m in models.get('models', [])):
        #     logger.warning(f"LLM model '{llm_model_name}' not found in Ollama. Please pull it.")

    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        ollama_client = None

    logger.info("Application startup complete.")
    yield # Application runs here
    # Shutdown logic
    logger.info("Shutting down application...")
    if mongo_client:
        logger.info("Closing MongoDB connection.")
        mongo_client.close()
    logger.info("Application shutdown complete.")


# Initialize FastAPI app with lifespan context manager
app = FastAPI(
    title="Local RAG Backend",
    description="API for the local RAG system with SvelteKit frontend.",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# --- CORS Middleware ---
# Define allowed origins (adjust as needed for production)
origins = [
    "http://localhost:5173", # SvelteKit dev server
    # Add other origins if needed (e.g., your production frontend URL)
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True, # Allows cookies to be included in requests
    allow_methods=["*"],    # Allows all methods (GET, POST, PUT, etc.)
    allow_headers=["*"],    # Allows all headers
)

# --- Dependency Functions ---
async def get_db() -> Union[motor.motor_asyncio.AsyncIOMotorDatabase, None]:
    """Dependency function to get the MongoDB database instance."""
    if db is None:
        # This ideally shouldn't happen if lifespan completes successfully,
        # but provides a safeguard.
        logger.error("MongoDB database instance is not available.")
        # Option 1: Raise an internal server error
        # raise HTTPException(status_code=500, detail="Database connection not initialized.")
        # Option 2: Return None and let the endpoint handle it (as it currently does)
        return None
    return db

async def get_embedding_model() -> SentenceTransformer: # Ensure model is returned or exception raised
    """Dependency function to get the SentenceTransformer model instance."""
    if embedding_model is None:
        logger.error("Embedding model instance is not available.")
        # Raise 503 if the model isn't loaded when the dependency is requested
        raise HTTPException(status_code=503, detail="Embedding model not loaded")
    return embedding_model

async def get_chroma_client() -> chromadb.HttpClient: # Ensure client is returned or exception raised
    """Dependency function to get the ChromaDB client instance."""
    if chroma_client is None:
        logger.error("ChromaDB client instance is not available.")
        # Raise 503 if the client isn't available when the dependency is requested
        raise HTTPException(status_code=503, detail="ChromaDB client not connected")
    return chroma_client

async def get_ollama_client() -> ollama.AsyncClient: # Ensure client is returned or exception raised
    """Dependency function to get the Ollama client instance."""
    if ollama_client is None:
        logger.error("Ollama client instance is not available.")
        # Raise 503 if the client isn't available when the dependency is requested
        raise HTTPException(status_code=503, detail="Ollama client not connected")
    return ollama_client

# --- API Endpoints ---

@app.get("/api/health", tags=["Health"])
async def health_check():
    """
    Simple health check endpoint.
    Returns the status of the backend service and its dependencies.
    """
    logger.debug("Health check endpoint called.")
    status = {
        "backend_status": "ok",
        "mongodb_status": "connected" if db is not None else "disconnected",
        "chromadb_status": "connected" if chroma_client is not None else "disconnected",
        "embedding_model_status": "loaded" if embedding_model is not None else "not_loaded",
        "ollama_status": "connected" if ollama_client is not None else "disconnected",
    }
    overall_ok = all(v in ["connected", "loaded", "ok"] for v in status.values())

    if overall_ok:
        return status
    else:
        # Return 503 Service Unavailable if any dependency failed
        raise HTTPException(status_code=503, detail=status)

# --- API Routers ---
from routers import documents, chat, dashboard # Import directly from routers subdir


app.include_router(documents.router, prefix="/api/documents", tags=["Documents"])
app.include_router(chat.router, prefix="/api/chat", tags=["Chat"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["Dashboard"]) # Add dashboard router

# Add a root endpoint for basic verification
@app.get("/", tags=["Root"])
async def read_root():
    logger.debug("Root endpoint called.")
    return {"message": "Welcome to the Local RAG Backend API"}


if __name__ == "__main__":
    # This block allows running the app directly with `python main.py`
    # Useful for local debugging without Docker/uvicorn command line
    import uvicorn
    logger.info("Running FastAPI app directly using uvicorn (debug mode)")
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True, # Enable auto-reload for development
        log_level=log_level.lower()
    )