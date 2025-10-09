from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from passlib.context import CryptContext
from app.db.session import get_db
from app.models.user import User
from app.schemas.user import UserCreate, UserOut


router = APIRouter(prefix="/users", tags=["users"])


pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")


@router.post("/", response_model=UserOut, status_code=status.HTTP_201_CREATED)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):

exists = db.query(User).filter(User.email == payload.email).first()
if exists:raise HTTPException(status_code=409, detail="Email zaten kayıtlı")

user = User(email=payload.email, password_hash=pwd.hash(payload.password))
db.add(user)
db.commit()
db.refresh(user)
return user


@router.get("/", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)):
return db.query(User).all()