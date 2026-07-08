from pathlib import Path

from semantic_media_search.ml.image_encoder import ImageEncoder
from semantic_media_search.ml.clip_model import ClipModel


image_path = Path("local_media/test.jpg")

print("Image path:", image_path)
print("Exists:", image_path.exists())
print("Is file:", image_path.is_file())

clip_model = ClipModel()

encoder = ImageEncoder(
    clip_model.model
)

embedding = encoder.encode(image_path)

print("Shape: ", embedding.shape)
print("Data type: ", embedding.dtype)
print("First values", embedding[:5])