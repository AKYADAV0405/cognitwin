import zipfile
from pathlib import Path


def package_extension() -> None:
    root_dir = Path(__file__).resolve().parent.parent
    dist_dir = root_dir / "apps" / "extension" / "dist"
    out_dir = root_dir / "apps" / "web" / "public" / "downloads"
    out_dir.mkdir(parents=True, exist_ok=True)
    zip_path = out_dir / "cognitwin-extension.zip"

    if not dist_dir.exists():
        print(f"Error: {dist_dir} does not exist. Run extension build first.")
        return

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in dist_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(dist_dir)
                zf.write(file_path, arcname)

    print(f"Successfully packaged extension to {zip_path}")


if __name__ == "__main__":
    package_extension()
