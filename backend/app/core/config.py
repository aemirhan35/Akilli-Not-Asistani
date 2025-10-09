from pydantic import BaseModel
from pydantic_settings import BaseSettings
import os


class Settings(BaseSettings):
APP_ENV: str = "dev"
API_PREFIX: str = "/api"
SECRET_KEY: str = "change_me"
ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
DB_URL: str = "sqlite:///./app.db"


class Config:
env_file = ".env"

settings = Settings()