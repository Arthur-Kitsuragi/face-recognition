from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    #.env
    lower_threshold: float = 30
    upper_threshold: float = 90
    area_factor: float = 0.3
    prompt_max_length: int = 50
    provider: str = "fal-ai"
    api_key: str = ""
    min_width: int = 720
    min_height: int = 405
    class Config:
        env_file = ".env"