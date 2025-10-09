from fastapi import FastAPI
from app.core.config import settings
from app.routers import users
from app.db.session import Base, engine

# İlk aşamada Alembic kullanmadan tabloları otomatik oluşturuyoruz
Base.metadata.create_all(bind=engine)

app = FastAPI(title="STT Notes API", version="0.1.0")
app.include_router(users.router, prefix=settings.API_PREFIX)

@app.get("/")
def root():
    return {"ok": True, "message": "Backend ayakta!"}
