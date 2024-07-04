from pydantic.v1 import BaseSettings


class Settings(BaseSettings):
    http_host: str
    http_port: int

    api_url: str
    api_auth_url: str
    api_login: str
    api_password: str

    class Config:
        env_file = ".env"


settings = Settings()
