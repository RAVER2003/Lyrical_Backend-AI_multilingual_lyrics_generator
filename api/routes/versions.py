from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from core.database import get_db
from api.dependencies import get_current_user
from models.user import User
from models.workspace import Workspace
from models.version import Version
from schemas.version import VersionResponse

router = APIRouter()

@router.get("/{workspace_id}/versions", response_model=List[VersionResponse])
def get_versions(workspace_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return db.query(Version).filter(Version.workspace_id == workspace_id).order_by(Version.timestamp.asc()).all()

@router.get("/{workspace_id}/versions/{version_id}", response_model=VersionResponse)
def get_version(workspace_id: str, version_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    version = db.query(Version).filter(Version.id == version_id, Version.workspace_id == workspace_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="Version not found")
    return version

@router.delete("/{workspace_id}/versions")
def delete_all_versions(workspace_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
        
    db.query(Version).filter(Version.workspace_id == workspace_id).delete()
    db.commit()
    return {"success": True}
