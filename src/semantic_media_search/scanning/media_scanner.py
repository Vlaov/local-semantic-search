from pathlib import Path


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
}


class MediaScanner:

    def scan_images(self, directory: Path) -> list[Path]:
        if not directory.exists():
            raise FileNotFoundError(
                f"Directory does not exist: {directory}"
            )

        if not directory.is_dir():
            raise NotADirectoryError(
                f"Path is not a directory: {directory}"
            )

        return sorted(
            path
            for path in directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )