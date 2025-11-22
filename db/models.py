from sqlalchemy import Column, Integer, Text, Boolean, DateTime
from sqlalchemy.sql import func

from .session import Base


class GenerationLog(Base):
    __tablename__ = "generation_logs"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    completion = Column(Text, nullable=False)
    is_correct = Column(Boolean, nullable=True)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


class SavedItem(Base):
    __tablename__ = "saved_items"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text, nullable=False)
    completion = Column(Text, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
