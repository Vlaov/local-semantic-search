from pathlib import Path

from semantic_media_search.indexing.vector_index import VectorIndex
from semantic_media_search.ml.clip_model import ClipModel
from semantic_media_search.ml.image_encoder import ImageEncoder
from semantic_media_search.ml.text_encoder import TextEncoder
from semantic_media_search.scanning.media_scanner import MediaScanner


clip_model = ClipModel()

scanner = MediaScanner()

image_encoder = ImageEncoder(
    clip_model.model
)

text_encoder = TextEncoder(
    clip_model.model
)

image_paths = scanner.scan_images(
    Path("local_media")
)

if not image_paths:
    raise RuntimeError("No images found")

image_embeddings = image_encoder.encode_batch(
    image_paths
)

vector_index = VectorIndex(
    dimension=image_embeddings.shape[1]
)

vector_index.add(
    image_embeddings
)

query = input("Search: ")

query_embedding = text_encoder.encode(
    query
)

scores, indices = vector_index.search(
    query_embedding,
    k=len(image_paths),
)

print()

for score, index in zip(scores, indices):
    print(
        f"{score:.4f} -> {image_paths[index]}"
    )