from typing import Optional

from pydantic import BaseModel


class QueryModel(BaseModel):
    message: str
    top_k: Optional[int] = 3
    min_similarity: Optional[float] = 0.7899
