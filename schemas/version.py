from pydantic import BaseModel
from datetime import datetime

class VersionBase(BaseModel):
    input_state: str
    output_state: str
    label: str

class VersionCreate(VersionBase):
    workspace_id: str

class VersionResponse(VersionBase):
    id: str
    workspace_id: str
    timestamp: datetime

    class Config:
        from_attributes = True
