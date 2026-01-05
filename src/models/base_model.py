from abc import ABC, abstractmethod
from typing import Any

class BaseTurnoverModel(ABC):
    def __init__(self):
        self.model = None

    @abstractmethod
    def fit(self, X, y) -> Any:
        pass

    @abstractmethod
    def predict_proba(self, X) -> Any:
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Any:
        pass
