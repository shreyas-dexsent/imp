"""Implementation for `tools.setup_ur5e_mjcf`."""

import json
import shutil
import zipfile
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

MENAGERIE_ZIPS = [
    "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/main.zip",
    "https://github.com/google-deepmind/mujoco_menagerie/archive/refs/heads/master.zip",
]


def download_zip(dest: Path) -> str:
    last_error = None
    for url in MENAGERIE_ZIPS:
        try:
            with urlopen(url) as resp:
                dest.write_bytes(resp.read())
            return url
        except HTTPError as exc:
            last_error = exc
    raise RuntimeError(f"Failed to download mujoco_menagerie zip: {last_error}")


def select_ur5e_xml(names: list[str]) -> str:
    candidates = [
        n for n in names if n.lower().endswith(".xml") and "ur5e" in n.lower()
    ]
    if not candidates:
        raise RuntimeError("UR5e MJCF not found in mujoco_menagerie zip")

    def score(name: str) -> tuple[int, int]:
        lower = name.lower()
        penalty = 0
        if "scene" in lower or "demo" in lower:
            penalty += 5
        if lower.endswith("/ur5e.xml"):
            penalty -= 5
        return (penalty, len(name))

    candidates.sort(key=score)
    return candidates[0]


def clear_model_root(model_root: Path, cache_name: str) -> None:
    for child in model_root.iterdir():
        if child.name == cache_name:
            continue
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    model_root = root / "robot_controller" / "assets" / "models" / "ur5e_mjcf"
    model_root.mkdir(parents=True, exist_ok=True)

    cache_name = ".cache_menagerie"
    cache = model_root / cache_name
    cache.mkdir(parents=True, exist_ok=True)
    zip_path = cache / "mujoco_menagerie.zip"

    if not zip_path.exists():
        print("Downloading MuJoCo menagerie...")
        source_url = download_zip(zip_path)
    else:
        print("Using cached MuJoCo menagerie zip.")
        source_url = MENAGERIE_ZIPS[0]

    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        xml_in_zip = select_ur5e_xml(names)
        model_dir = Path(xml_in_zip).parent.as_posix() + "/"
        members = [m for m in names if m.startswith(model_dir)]
        zf.extractall(cache, members)

    extracted_model_dir = cache / Path(model_dir)
    if not extracted_model_dir.exists():
        raise RuntimeError(f"Extracted model dir missing: {extracted_model_dir}")

    print("Copying UR5e MJCF model...")
    clear_model_root(model_root, cache_name)
    shutil.copytree(extracted_model_dir, model_root, dirs_exist_ok=True)

    model_xml = model_root / Path(xml_in_zip).name
    if not model_xml.exists():
        raise RuntimeError(f"UR5e MJCF not found after copy: {model_xml}")

    # Provide a stable name for configs.
    stable_xml = model_root / "ur5e.xml"
    if model_xml.name != stable_xml.name:
        shutil.copyfile(model_xml, stable_xml)

    meta = {
        "source": source_url,
        "xml_in_zip": xml_in_zip,
        "model_xml": str(stable_xml),
    }
    (model_root / "ur5e_source.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    print(f"Done. Model ready at {stable_xml}")


if __name__ == "__main__":
    main()
