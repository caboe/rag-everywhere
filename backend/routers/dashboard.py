# backend/routers/dashboard.py
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
import logging
import motor.motor_asyncio # Add motor import for type hint

# Assuming main.py defines db
from main import logger, get_db # Import logger and get_db

router = APIRouter()

class DashboardStats(BaseModel):
    total_documents: int
    total_folders: int
    # Add more stats later if needed (e.g., db size, selected count)

@router.get("/stats", response_model=DashboardStats)
async def get_dashboard_stats(db: motor.motor_asyncio.AsyncIOMotorDatabase = Depends(get_db)): # Inject DB
    """
    Retrieves basic statistics for the dashboard.
    """
    if db is None: # Corrected check
        raise HTTPException(status_code=503, detail="MongoDB connection not available.")

    try:
        logger.info("Fetching dashboard statistics...")
        # Count documents (items that are not folders)
        doc_count = await db["documents"].count_documents({"is_folder": False})
        # Count folders (items that are folders)
        folder_count = await db["documents"].count_documents({"is_folder": True})
        logger.info(f"Stats fetched: Documents={doc_count}, Folders={folder_count}")

        return DashboardStats(
            total_documents=doc_count,
            total_folders=folder_count
        )
    except Exception as e:
        logger.error(f"Failed to retrieve dashboard stats from MongoDB: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve dashboard statistics.")