from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field

try:
	from pydantic_settings import BaseSettings, SettingsConfigDict

	_PYDANTIC_V2 = True
except ImportError:
	from pydantic import BaseSettings

	_PYDANTIC_V2 = False


class Settings(BaseSettings):
	DATA_PATH: Path
	MODEL_PATH: Path
	API_KEY: str = Field(..., min_length=1)

	if _PYDANTIC_V2:
		model_config = SettingsConfigDict(
			env_file=".env",
			env_file_encoding="utf-8",
			extra="ignore",
		)
	else:

		class Config:
			env_file = ".env"
			env_file_encoding = "utf-8"
			case_sensitive = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
	return Settings()


settings = get_settings()

