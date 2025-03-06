from abc import ABC, abstractmethod
import pandas as pd

class Cache(ABC):
    """Interface for caching frequently accessed data."""
    
    @abstractmethod
    def get(self, key: str) -> pd.DataFrame:
        """Retrieve data from cache."""
        pass
    
    @abstractmethod
    def set(self, key: str, value: pd.DataFrame) -> None:
        """Store data in cache."""
        pass