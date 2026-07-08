from pathlib import Path

from semantic_media_search.ml import clip_model
from semantic_media_search.ml.clip_model import ClipModel
from semantic_media_search.ml.image_encoder import ImageEncoder
from semantic_media_search.scanning.media_scanner import MediaScanner

clip_model = ClipModel()

scanner = MediaScanner()
encoder = ImageEncoder(
    clip_model.model
)

image_paths = scanner.scan_images(
    Path("local_media")
)

if not image_paths:
    raise RuntimeError("No images found")

embeddings = encoder.encode_batch(image_paths)

for index, image_path in enumerate(image_paths):
    print(f"{index}: {image_path}")

print(f"Embeddings shape: {embeddings.shape}")
print(f"Data type: {embeddings.dtype}")