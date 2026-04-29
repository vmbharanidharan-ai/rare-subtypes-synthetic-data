from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, context: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
