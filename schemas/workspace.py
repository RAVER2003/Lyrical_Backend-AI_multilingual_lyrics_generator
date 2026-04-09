from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class WorkspaceBase(BaseModel):
    title: str
    preview: Optional[str] = None
    current_input: str = ""
    current_output: str = ""
    notes: List[str] = []

class WorkspaceCreate(BaseModel):
    title: str = "New Translation"

class WorkspaceUpdate(BaseModel):
    title: Optional[str] = None
    current_input: Optional[str] = None
    current_output: Optional[str] = None
    notes: Optional[List[str]] = None

class WorkspaceResponse(WorkspaceBase):
    id: str
    user_id: str
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True
