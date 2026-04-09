from sqlalchemy import Column, String, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from datetime import datetime
from core.database import Base
import uuid

class Version(Base):
    __tablename__ = "versions"
    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    workspace_id = Column(String, ForeignKey("workspaces.id", ondelete="CASCADE"))
    input_state = Column(Text)
    output_state = Column(Text)
    label = Column(String)
    timestamp = Column(DateTime, default=datetime.utcnow)

    workspace = relationship("Workspace")
