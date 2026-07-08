from pathlib import Path

from semantic_media_search.indexing.vector_index import VectorIndex
from semantic_media_search.ml.clip_model import ClipModel
from semantic_media_search.ml.image_encoder import ImageEncoder
from semantic_media_search.ml.text_encoder import TextEncoder
from semantic_media_search.scanning.media_scanner import MediaScanner
from semantic_media_search.search.search_engine import SearchEngine


def main() -> None:
    clip_model = ClipModel()

    image_encoder = ImageEncoder(
        clip_model.model
    )

    text_encoder = TextEncoder(
        clip_model.model
    )

    scanner = MediaScanner()

    image_paths = scanner.scan_images(
        Path("local_media")
    )

    if not image_paths:
        raise RuntimeError(
            "No images found"
        )

    image_embeddings = image_encoder.encode_batch(
        image_paths
    )

    vector_index = VectorIndex(
        dimension=image_embeddings.shape[1]
    )

    vector_index.add(
        image_embeddings
    )

    search_engine = SearchEngine(
        text_encoder=text_encoder,
        vector_index=vector_index,
        media_paths=image_paths,
    )

    while True:
        query = input("Search: ").strip()

        if not query:
            break

        results = search_engine.search(
            query,
            limit=5,
        )

        print()

        for position, result in enumerate(
            results,
            start=1,
        ):
            print(
                f"{position}. "
                f"{result.score:.4f} -> "
                f"{result.path}"
            )

        print()


if __name__ == "__main__":
    main()