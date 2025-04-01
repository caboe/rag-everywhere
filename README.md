# RAG Everywhere

## Purpose

RAG Everywhere is a totally autonomous, local Retrieval-Augmented Generation (RAG) system. It allows you to chat with your documents locally, without relying on external cloud services.

## Features

*   **Local First:** Runs entirely on your local machine using Docker.
*   **Document Management:** Upload and manage your documents for the RAG process.
*   **Chat Interface:** Interact with your documents through a user-friendly chat interface.
*   **Configurable:** (Assuming configuration options exist based on settings routes)

## Setup Instructions

Follow these steps to set up and run the project after cloning the repository.

### Prerequisites

*   [Docker](https://docs.docker.com/get-docker/)
*   [Docker Compose](https://docs.docker.com/compose/install/)
*   [Bun](https://bun.sh/docs/installation) (Primarily for potential local frontend development/tasks, though Docker handles the main setup)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url> # Replace <repository-url> with the actual URL
    cd rag-everywhere
    ```

2.  **Environment Variables:**
    Create a `.env` file in the project root directory. You may need to copy an example file if one is provided (e.g., `.env.example`) or configure it based on the required variables for the backend and frontend services. (Check `docker-compose.yml` and application code for required variables).

3.  **Build and Run with Docker Compose:**
    This command will build the Docker images for the frontend and backend services and start the containers.
    ```bash
    docker-compose up --build -d
    ```
    *(Using `-d` to run in detached mode)*

4.  **Access the Application:**
    Once the containers are running, you should be able to access the frontend application in your web browser. The default URL is typically:
    *   `http://localhost:5173` (Common for Vite/SvelteKit development servers)
    *(Check the `docker-compose.yml` file for the exact port mapping if this doesn't work).*

## Usage

1.  Open the application in your browser.
2.  Navigate to the upload section to add your documents.
3.  Go to the chat section to start interacting with your documents.

## Development (Optional)

If you want to run the frontend or backend services locally outside of Docker (e.g., for development):

### Frontend (SvelteKit with Bun)

```bash
cd frontend
bun install
bun run dev
```

### Backend (Python/FastAPI)

```bash
cd backend
# Setup a Python virtual environment (recommended)
python -m venv venv
source venv/bin/activate # or venv\Scripts\activate on Windows
pip install -r requirements.txt
# Run the FastAPI server (check main.py for the command, likely uvicorn)
uvicorn main:app --reload --host 0.0.0.0 --port 8000