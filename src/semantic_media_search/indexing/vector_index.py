import faiss
import numpy as np
from numpy.typing import NDArray


class VectorIndex:

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)

    @property
    def size(self) -> int:
        return int(self._index.ntotal)

    def add(
        self,
        embeddings: NDArray[np.float32],
    ) -> None:
        vectors = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if vectors.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2-dimensional array"
            )

        if vectors.shape[1] != self._dimension:
            raise ValueError(
                "Embedding dimension does not match index dimension"
            )

        vectors = vectors.copy()

        faiss.normalize_L2(vectors)

        self._index.add(vectors)

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 5,
    ) -> tuple[
        NDArray[np.float32],
        NDArray[np.int64],
    ]:
        if k <= 0:
            raise ValueError("k must be greater than zero")

        if self._index.ntotal == 0:
            raise RuntimeError("Vector index is empty")

        query = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2:
            raise ValueError(
                "Query embedding must be a 1D or 2D array"
            )

        if query.shape[1] != self._dimension:
            raise ValueError(
                "Query dimension does not match index dimension"
            )

        query = query.copy()

        faiss.normalize_L2(query)

        scores, indices = self._index.search(
            query,
            min(k, self._index.ntotal),
        )

        return scores[0], indices[0]