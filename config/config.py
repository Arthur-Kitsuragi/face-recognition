from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    #.env
    lower_threshold: float = 30
    upper_threshold: float = 90
    area_factor: float = 0.3
    class Config:
        env_file = ".env"