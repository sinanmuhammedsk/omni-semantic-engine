import os
import streamlit as st
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base
from ..core.config import settings

# FORCE an in-memory database string if running on Streamlit Cloud to bypass any Pydantic caching bugs
if st.secrets.get("GROQ_API_KEY"):
    db_url = "sqlite:///:memory:"
else:
    db_url = settings.DATABASE_URL

engine = create_engine(
    db_url, 
    connect_args={"check_same_thread": False} if db_url.startswith("sqlite") else {}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    from .models import DocumentMetadata
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
