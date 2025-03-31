# backend/models/mongo_models.py
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime
from bson import ObjectId # For MongoDB compatibility

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, _info): # Add _info to accept the third argument
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, field_schema):
        field_schema.update(type="string")


class DocumentMetadata(BaseModel):
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    filename: str = Field(...)
    filepath: str = Field(...) # Path to the original file in the volume
    mimetype: Optional[str] = None
    upload_date: datetime = Field(default_factory=datetime.utcnow)
    parent_id: Optional[str] = None # ID of the parent folder in the tree (or None for root)
    is_folder: bool = Field(default=False)
    selected_for_rag: bool = Field(default=True) # Default to selected when uploaded? Or False? Let's start with True.
    processing_status: str = Field(default="queued") # e.g., queued, processing, completed, failed, unsupported
    updated_at: datetime = Field(default_factory=datetime.utcnow) # Add updated timestamp
    # Add other fields as needed, e.g., size, owner

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True # Needed for ObjectId
        json_encoders = {ObjectId: str} # Convert ObjectId to str for JSON responses