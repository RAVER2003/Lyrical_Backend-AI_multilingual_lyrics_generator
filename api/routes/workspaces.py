from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from core.database import get_db
from api.dependencies import get_current_user
from models.user import User
from models.workspace import Workspace
from schemas.workspace import WorkspaceCreate, WorkspaceUpdate, WorkspaceResponse

router = APIRouter()

@router.get("/", response_model=List[WorkspaceResponse])
def get_workspaces(db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    return db.query(Workspace).filter(Workspace.user_id == current_user.id).all()

@router.post("/", response_model=WorkspaceResponse, status_code=status.HTTP_201_CREATED)
def create_workspace(workspace_in: WorkspaceCreate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    new_workspace = Workspace(
        user_id=current_user.id,
        title=workspace_in.title
    )
    db.add(new_workspace)
    db.commit()
    db.refresh(new_workspace)
    return new_workspace

@router.get("/{workspace_id}", response_model=WorkspaceResponse)
def get_workspace(workspace_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    return workspace

@router.patch("/{workspace_id}")
def update_workspace(workspace_id: str, workspace_update: WorkspaceUpdate, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    update_data = workspace_update.dict(exclude_unset=True)
    for key, value in update_data.items():
        setattr(workspace, key, value)
        
    db.commit()
    return {"success": True}

@router.delete("/{workspace_id}")
def delete_workspace(workspace_id: str, db: Session = Depends(get_db), current_user: User = Depends(get_current_user)):
    workspace = db.query(Workspace).filter(Workspace.id == workspace_id, Workspace.user_id == current_user.id).first()
    if not workspace:
        raise HTTPException(status_code=404, detail="Workspace not found")
    
    # Cascade delete is handled by DB config in models, or we explicitly delete versions if needed.
    # Since we set ondelete="CASCADE" in Version model, standard delete works.
    db.delete(workspace)
    db.commit()
    return {"success": True}
