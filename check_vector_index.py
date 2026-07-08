from pathlib import Path

from semantic_media_search.indexing.vector_index import VectorIndex
from semantic_media_search.ml.image_encoder import ImageEncoder
from semantic_media_search.scanning.media_scanner import MediaScanner
from semantic_media_search.ml.clip_model import ClipModel



scanner = MediaScanner()

clip_model = ClipModel()

encoder = ImageEncoder(
    clip_model.model
)

image_paths = scanner.scan_images(
    Path("local_media")
)

if not image_paths:
    raise RuntimeError("No images found")

embeddings = encoder.encode_batch(image_paths)

vector_index = VectorIndex(
    dimension=embeddings.shape[1]
)

vector_index.add(embeddings)

scores, indices = vector_index.search(
    embeddings[0],
    k=len(image_paths),
)

for score, index in zip(scores, indices):
    print(
        f"{score:.4f} -> {image_paths[index]}"
    )