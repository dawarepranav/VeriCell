"""Assemble the 69-image MorphoOrgaVQA image set from its three public source datasets.

MorphoOrgaVQA (Zhang & Hoermann, arXiv 2609.08696) is built on the test sets of OrganoID,
OrgaExtractor and OrgaSegment, 69 images in total. The paper doesn't list the files, so this
script reconstructs the split:
  - OrgaSegment eval set (12)            Zenodo 10.5281/zenodo.10278229 (instance masks)
  - OrgaExtractor test set (10)          Google Drive folder from github.com/tpark16/orgaextractor (binary)
  - OrganoID testing (28 + 4) and the
    OriginalData validation set (15)     OSF osf.io/xmes4 (binary)
12 + 10 + 47 = 69. Binary masks become instance masks via 8-connected components, as the
benchmark requires instance labels.

Output: data/morphoorga/{images/<name>.png, masks/<name>.npy}
"""

import io
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import httpx
import tifffile
from PIL import Image
from remotezip import RemoteZip

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "morphoorga" / "raw"
OUT = ROOT / "data" / "morphoorga"
ORGASEGMENT = "https://zenodo.org/api/records/10278229/files/OrganoidBasic_v20211206.zip/content"
ORGAEXTRACTOR = "https://drive.google.com/drive/folders/17K4N7gEZUqAcwf9N2-I5DPbywwPvzAvo"
ORGANOID = "https://files.osf.io/v1/resources/xmes4/providers/osfstorage/62e2e02827b74638320ac46d"
ORGANOID_SPLITS = ["OriginalData/testing", "MouseOrganoids/testing", "OriginalData/validation"]


def read_any(data: bytes) -> np.ndarray:
    try:
        return tifffile.imread(io.BytesIO(data))
    except Exception:
        return np.asarray(Image.open(io.BytesIO(data)))


def instances(binary: np.ndarray) -> np.ndarray:
    if binary.ndim == 3:
        binary = binary[..., 0]
    _, labels = cv2.connectedComponents((binary > 0).astype(np.uint8), connectivity=8)
    return labels.astype(np.int32)


def save(name: str, image: np.ndarray, mask: np.ndarray) -> None:
    if image.ndim == 3 and image.shape[2] == 4:
        image = image[..., :3]
    Image.fromarray(image).save(OUT / "images" / f"{name}.png")
    np.save(OUT / "masks" / f"{name}.npy", mask.astype(np.int32))


def main() -> None:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    (OUT / "masks").mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    n = 0

    # OrgaSegment: instance-labelled PNG masks next to JPG images.
    seg_dir = RAW / "orgasegment"
    if not seg_dir.exists():
        (RAW / "orgasegment.zip").write_bytes(httpx.get(ORGASEGMENT, follow_redirects=True, timeout=300).content)
        subprocess.run(["unzip", "-q", "-o", str(RAW / "orgasegment.zip"), "-d", str(seg_dir)], check=True)
    for img in sorted(seg_dir.glob("*/eval/*_img.jpg")):
        mask = np.asarray(Image.open(str(img).replace("_img.jpg", "_masks_organoid.png")))
        save(f"orgasegment_{img.stem.replace('_img', '')}", np.asarray(Image.open(img)), mask)
        n += 1

    # OrgaExtractor: binary TIFF labels.
    ex_dir = RAW / "orgaextractor"
    if not (ex_dir / "test").exists():
        subprocess.run([sys.executable, "-m", "gdown", "--folder", "-q", ORGAEXTRACTOR, "-O", str(ex_dir)], check=True)
        subprocess.run(["unzip", "-q", "-o", str(ex_dir / "test.zip"), "-d", str(ex_dir / "test")], check=True)
    for img in sorted((ex_dir / "test").rglob("input_*.jpg")):
        label = tifffile.imread(img.with_name(img.name.replace("input_", "label_").replace(".jpg", ".tif")))
        save(f"orgaextractor_{img.stem.replace('input_', '')}", np.asarray(Image.open(img)), instances(label))
        n += 1

    # OrganoID: binary masks with the same file name as the image, pulled out of the 3.6 GB zip.
    with RemoteZip(ORGANOID) as z:
        names = set(z.namelist())
        for split in ORGANOID_SPLITS:
            for path in sorted(p for p in names if p.startswith(f"{split}/images/") and not p.endswith("/")):
                seg = path.replace("/images/", "/segmentations/")
                if seg not in names:
                    print(f"skip {path}: no segmentation", flush=True)
                    continue
                tag = split.split("/")[0].replace("OriginalData", "orig").replace("MouseOrganoids", "mouse")
                name = f"organoid_{tag}_{split.split('/')[1][:4]}_{Path(path).stem}"
                image = read_any(z.read(path))
                if image.dtype != np.uint8:
                    lo, hi = np.percentile(image, (0.5, 99.5))
                    image = np.clip((image - lo) / (hi - lo + 1e-6) * 255, 0, 255).astype(np.uint8)
                save(name, image, instances(read_any(z.read(seg))))
                n += 1
    print(f"done: {n} images -> {OUT}")


if __name__ == "__main__":
    main()
