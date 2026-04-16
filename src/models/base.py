from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
	"""Abstract base class for all trainable models in this project."""

	def __init__(self, model_name: str | None = None) -> None:
		self.model_name = model_name or self.__class__.__name__

	@abstractmethod
	def train(self, X: Any, y: Any | None = None, **kwargs: Any) -> None:
		"""Train the model using input features and optional labels."""

	@abstractmethod
	def predict(self, X: Any, **kwargs: Any) -> Any:
		"""Run inference and return model predictions."""

	@abstractmethod
	def evaluate(self, X: Any, y: Any, **kwargs: Any) -> dict[str, float]:
		"""Evaluate model performance and return metric dictionary."""

	@abstractmethod
	def save(self, path: str | Path, **kwargs: Any) -> None:
		"""Persist the trained model to disk."""

	@classmethod
	@abstractmethod
	def load(cls, path: str | Path, **kwargs: Any) -> BaseModel:
		"""Load a model from disk and return a model instance."""


__all__ = ["BaseModel"]

