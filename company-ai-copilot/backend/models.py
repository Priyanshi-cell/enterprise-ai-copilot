import time
from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy.sql import func

from core.db import Base


class ChatHistory(Base):
    __tablename__ = "chat_history"

    id = Column(Integer, primary_key=True, index=True)
    company = Column(String(255), nullable=False, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    retrieved_docs = Column(Text, nullable=True)   
    retrieval_latency_ms = Column(Float, nullable=True)
    generation_latency_ms = Column(Float, nullable=True)
    total_latency_ms = Column(Float, nullable=True)
    top_rerank_score = Column(Float, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())