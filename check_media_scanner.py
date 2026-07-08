from pathlib import Path

from semantic_media_search.scanning.media_scanner import (
    MediaScanner,
)


scanner = MediaScanner()

images = scanner.scan_images(
    Path("local_media")
)

for image in images:
    print(image)

print(f"Found images: {len(images)}")