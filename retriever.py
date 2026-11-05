from typing import Optional

import config
import store


def retrieve(col, embeddings, question: str, k: Optional[int] = None ) -> list[dict]:

    vector = embeddings.embed_query(question)

    return store.hybrid_search(
        col, question, vector, limit = k or config.TOP_K, alpha = config.ALPHA
    )

def label(chunk: dict) -> str:

    page = chunk.get('page', -1)

    return chunk['source'] + (f', p.{page + 1} ' if page >= 0 else '')
