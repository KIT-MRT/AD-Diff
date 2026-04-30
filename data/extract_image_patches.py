import json
import logging
import os
from datetime import datetime
from abc import ABC, abstractmethod
try:
    from enum import StrEnum
except ImportError: # Python < 3.11, we cannot use newer python versions with waymo
    from enum import Enum
    class StrEnum(str, Enum):
        pass
import datetime
import zoneinfo
import argparse
from typing import Iterator, List , Tuple
from io import BytesIO
from pathlib import Path

from pydantic import BaseModel, PositiveInt, ConfigDict

from PIL import Image, ImageDraw
from tqdm import tqdm

repo_root = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..")
with open(os.path.join(repo_root, "data_dir.json"), "r") as f:
    json_data = json.load(f)
data_dir = json_data["ad_data_dir"]
if not data_dir:
    data_dir = os.path.join(repo_root, "data", "ad-datasets")

def crop(image: Image.Image, bbox: list[int], pad_factor: float = 0.0, fill_color: tuple[int, int, int]|None = None) -> Tuple[Image.Image, List[int]]:
    """
    Crops a bounding box from the image, with optional padding.
    Args:
        image: PIL Image to crop from.
        bbox: Bounding box [x_min, y_min, x_max, y_max].
        image_width: Width of the original image.
        image_height: Height of the original image.
        pad_factor: Fractional padding to add to each side of the bbox.
        fill_color: Color (RGB) to use for padding areas outside the image. If None, the returned patch is clipped to the image boundaries.
        Returns:
            Cropped PIL Image patch, top left coordinates of the patch in the original image (x_min, y_min).
            x_min, y_min, x_max, y_max = bbox
    """
    x_min, y_min, x_max, y_max = bbox
    bbox_width = x_max - x_min
    bbox_height = y_max - y_min

    pad_w = int(bbox_width * pad_factor)
    pad_h = int(bbox_height * pad_factor)

    crop_x_min = x_min - pad_w
    crop_y_min = y_min - pad_h
    crop_x_max = x_max + pad_w
    crop_y_max = y_max + pad_h

    if fill_color is not None:
        # Create a new image with the desired size and fill color
        new_width = (crop_x_max - crop_x_min)
        new_height = (crop_y_max - crop_y_min)
        new_image = Image.new(image.mode, (new_width, new_height), fill_color)

        # Calculate the position to paste the original image onto the new image
        paste_x = max(0, -crop_x_min)
        paste_y = max(0, -crop_y_min)

        # Calculate the region of the original image to paste
        src_x_min = max(0, crop_x_min)
        src_y_min = max(0, crop_y_min)
        src_x_max = min(image.width, crop_x_max)
        src_y_max = min(image.height, crop_y_max)

        # Paste the relevant region of the original image onto the new image
        new_image.paste(image.crop((src_x_min, src_y_min, src_x_max, src_y_max)), (paste_x, paste_y))
        return new_image, [crop_x_min, crop_y_min]
    else:
        # Clip the crop coordinates to be within the image boundaries
        crop_x_min = max(0, crop_x_min)
        crop_y_min = max(0, crop_y_min)
        crop_x_max = min(image.width, crop_x_max)
        crop_y_max = min(image.height, crop_y_max)

        return image.crop((crop_x_min, crop_y_min, crop_x_max, crop_y_max)), [crop_x_min, crop_y_min]

def paint_bbox(image: Image.Image, bbox: list[int], color: tuple[int, int, int], thickness: int = 5) -> Image.Image:
    """
    Paints a bounding box on the image.
    Args:
        image: PIL Image to paint on.
        bbox: Bounding box [x_min, y_min, x_max, y_max].
        color: Color (RGB) to use for the bounding box.
        thickness: Thickness of the bounding box lines.
    Returns:
        PIL Image with the bounding box painted.
    """
    draw = ImageDraw.Draw(image)
    x_min, y_min, x_max, y_max = bbox
    for t in range(thickness):
        draw.rectangle([x_min - t, y_min - t, x_max + t, y_max + t], outline=color)
    return image

def _downscale_full_image(image: Image.Image, max_size: int = 512) -> Image.Image:
    """Return a downscaled copy of the full image, preserving aspect ratio.

    The longer side is limited to max_size pixels; smaller images are left unchanged.
    """
    width, height = image.size
    if width <= max_size and height <= max_size:
        return image
    scale = min(max_size / float(width), max_size / float(height))
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    # Prefer modern Resampling enum when available, fall back otherwise.
    try:  # pragma: no cover - compatibility shim
        resample = Image.Resampling.BILINEAR
    except AttributeError:  # older Pillow versions
        resample = Image.BILINEAR
    return image.resize((new_width, new_height), resample=resample)


class DatasetSplit(StrEnum):
    TRAIN = "train"
    VAL = "val"
    TRAINVAL = "trainval"
    TEST = "test"
    MINI = "mini"

class PatchData(BaseModel):
    original_image_path: str
    bbox_width: int
    bbox_height: int
    bbox_top_left: list[int]
    location: str
    category: str | None
    patch_width: int | None = None # equal to bbox_width if None (if no padding is applied)
    patch_height: int | None = None # equal to bbox_height if None (if no padding is applied)
    patch_top_left: list[int] | None = None # equal to bbox_top_left if None (if no padding is applied)
    sub_categories: list[str] | None = None
    attributes: dict[str,str] | None = None # {attribute_name: attribute_description}
    log: str | None = None
    vehicle: str | None = None
    local_datetime: datetime.datetime | None = None
    camera: str | None = None
    weather: str | None = None
    speed: float | None = None
    acceleration: list[float | None] | None = None
    yaw_rate: float | None = None

    model_config = ConfigDict(extra='forbid', strict=True)

class PatchEntry(BaseModel):
    patchpath: str
    patch_data: PatchData

    model_config = ConfigDict(extra='forbid', strict=True)

class PatchDataset(BaseModel):
    dataset_name: str
    class_name: str
    split: DatasetSplit
    min_patch_size: PositiveInt = 50
    padding_factor: float = 0.0
    padding_color: tuple[int, int, int] | None = None
    bbox_color: tuple[int, int, int] | None = None
    bbox_thickness: int = 5
    patches: list[PatchEntry]

    model_config = ConfigDict(extra='forbid', strict=True)

class PatchIterator(ABC):
    def __init__(self, root: str, dataset_name: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        self.root = root
        self.dataset_name = dataset_name
        self.class_name = class_name
        self.min_patch_size = min_patch_size
        self.split = split
        self.patch_pad_factor = patch_pad_factor
        self.patch_fill_color = patch_fill_color
        self.paint_bboxes = paint_bboxes
        self.bbox_color = bbox_color
        self.bbox_thickness = bbox_thickness

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __iter__(self) -> Iterator[tuple[Image, PatchData]]:
        pass

class NuImagesIterator(PatchIterator):
    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, dataset_name="nuimages", class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)


    def _get_super_category_from_class_name(self, class_name: str) -> str:
        """
        Extracts the super category from the class name.
        Assumes that the class name is formatted as 'super_category.class_name'.
        """
        return class_name.split('.', 1)[0]

    def _get_sub_categories_from_class_name(self, class_name: str) -> List[str]:
        """
        Extracts the sub-categories from the class name.
        Assumes that the class name is formatted as 'super_category.class_name'.
        Returns an empty list if there are no sub-categories.
        """
        parts = class_name.split('.')
        return parts[1:] if len(parts) > 1 else []

    def _get_timezone_from_log(self, log: dict) -> zoneinfo.ZoneInfo:
        if log["location"].lower().startswith("singapore"):
            return zoneinfo.ZoneInfo("Asia/Singapore")
        elif log["location"].lower().startswith("boston"):
            return zoneinfo.ZoneInfo("America/New_York")
        else:
            raise ValueError(f"Unknown location '{log['location']}' for timezone conversion.")

    def _get_datetime(self, sample_data: dict, log: dict) -> datetime.datetime:
        timezone = self._get_timezone_from_log(log)
        timestamp = sample_data["timestamp"]
        timestamp_seconds = timestamp / 1e6  # Convert microseconds to seconds
        date_time = datetime.datetime.fromtimestamp(timestamp_seconds, tz=timezone)
        return date_time


class NuImagesPatchIterator(NuImagesIterator):
    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)
        from nuimages import NuImages

        self.nuimages = NuImages(version=f"v1.0-{split}", dataroot=root, lazy=False, verbose=True)
        self.category_tokens = [cat["token"] for cat in self.nuimages.category if self._get_super_category_from_class_name(cat["name"]) == class_name] # nuImages separates classes and sub-classes with a dot
        assert self.category_tokens, f"Class '{class_name}' not found in NuImages categories."
        self.len = sum(
            1 for ann in self.nuimages.object_ann if ann["category_token"] in self.category_tokens
        )

    def __len__(self):
        return self.len

    def __iter__(self):
        for ann in self.nuimages.object_ann:
            if not ann["category_token"] in self.category_tokens:
                continue
            x_min, y_min, x_max, y_max = ann["bbox"]
            bbox_width = x_max - x_min
            bbox_height = y_max - y_min
            if bbox_width < self.min_patch_size or bbox_height < self.min_patch_size:
                continue

            instance = self.nuimages.get("sample_data", ann["sample_data_token"])
            sample = self.nuimages.get("sample", instance["sample_token"])
            image_path = os.path.join(self.root, instance["filename"])
            image = Image.open(image_path).convert("RGB")
            if self.paint_bboxes:
                image = paint_bbox(image, [int(x_min), int(y_min), int(x_max), int(y_max)], color=self.bbox_color, thickness=self.bbox_thickness)
            patch, patch_top_left = crop(
                image,
                bbox=[int(x_min), int(y_min), int(x_max), int(y_max)],
                pad_factor=self.patch_pad_factor,
                fill_color=self.patch_fill_color
            )
            class_name = self.nuimages.get("category", ann["category_token"])["name"]
            log = self.nuimages.get("log", sample["log_token"])
            ego_pose = self.nuimages.get("ego_pose", instance["ego_pose_token"])
            patch_data = PatchData(
                original_image_path=instance["filename"],
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                bbox_top_left=[x_min, y_min],
                patch_width=patch.width,
                patch_height=patch.height,
                patch_top_left=patch_top_left,
                location=log["location"],
                category=self.class_name,
                sub_categories=self._get_sub_categories_from_class_name(class_name),
                attributes={self.nuimages.get("attribute", attribute_token)["name"]: self.nuimages.get("attribute", attribute_token)["description"] for attribute_token in ann["attribute_tokens"]},
                log=log["logfile"],
                vehicle=log["vehicle"],
                local_datetime=self._get_datetime(instance, log),
                camera=self.nuimages.get("calibrated_sensor", instance["calibrated_sensor_token"])["sensor_token"],
                weather=None,  # nuImages does not provide weather information in the object annotations
                speed=ego_pose["speed"],
                acceleration=ego_pose["acceleration"],
                yaw_rate=ego_pose["rotation_rate"][2]
            )
            yield patch, patch_data


class NuImagesImageIterator(NuImagesIterator):
    """Iterate over all nuImages camera frames.

    Yields a single, downscaled patch covering the whole image per frame.
    Images are not duplicated. Instance-specific fields (category,
    sub_categories, attributes) are set to None; bbox spans the full
    (downscaled) image.
    """

    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)
        from nuimages import NuImages

        self.nuimages = NuImages(version=f"v1.0-{split}", dataroot=root, lazy=False, verbose=True)

        # select keyframes
        self.sample_data_tokens = [
            s["key_camera_token"] for s in self.nuimages.sample
        ]
        self.len = len(self.sample_data_tokens)

    def __len__(self):
        return self.len

    def __iter__(self):
        for token in self.sample_data_tokens:
            instance = self.nuimages.get("sample_data", token)
            sample = self.nuimages.get("sample", instance["sample_token"])
            image_path = os.path.join(self.root, instance["filename"])
            image = Image.open(image_path).convert("RGB")
            patch = _downscale_full_image(image)
            bbox_width, bbox_height = patch.size
            log = self.nuimages.get("log", sample["log_token"])
            ego_pose = self.nuimages.get("ego_pose", instance["ego_pose_token"])
            patch_data = PatchData(
                original_image_path=instance["filename"],
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                bbox_top_left=[0, 0],
                location=log["location"],
                category=None,
                sub_categories=None,
                attributes=None,
                log=log["logfile"],
                vehicle=log["vehicle"],
                local_datetime=self._get_datetime(instance, log),
                camera=self.nuimages.get("calibrated_sensor", instance["calibrated_sensor_token"])["sensor_token"],
                weather=None,
                speed=ego_pose["speed"],
                acceleration=ego_pose["acceleration"],
                yaw_rate=ego_pose["rotation_rate"][2],
            )
            yield patch, patch_data



class KittiIterator(PatchIterator):
    KITTI_CLASSES = [
        "Car", "Van", "Truck", "Pedestrian", "Person_sitting",
        "Cyclist", "Tram", "Misc", "DontCare"
    ]
    # KITTI 2D object detection dataset: https://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=2d

    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, dataset_name="kitti", class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)
        if split in (DatasetSplit.TRAINVAL, DatasetSplit.TRAIN):
            dataset_root = os.path.join(root, "training")
        elif split == DatasetSplit.TEST:
            raise ValueError(f"KITTI test split does not have labels available.")
        else:
            raise ValueError(f"Unsupported split '{split}' for KITTI dataset.")

        self.image_dir = os.path.join(dataset_root, "image_2")
        self.label_dir = os.path.join(dataset_root, "label_2")
        self.image_files = sorted([
            f for f in os.listdir(self.image_dir)
            if f.endswith(".png") or f.endswith(".jpg")
        ])
        self.label_files = sorted([
            f for f in os.listdir(self.label_dir)
            if f.endswith(".txt")
        ])
        # Only keep images that have a corresponding label file
        self.samples = [
            (img, img.replace(".png", ".txt").replace(".jpg", ".txt"))
            for img in self.image_files
            if img.replace(".png", ".txt").replace(".jpg", ".txt") in self.label_files
        ]
        # Precompute length
        self._patch_indices = []
        for img_file, label_file in self.samples:
            label_path = os.path.join(self.label_dir, label_file)
            with open(label_path, "r") as f:
                for line in f:
                    fields = line.strip().split()
                    if len(fields) < 15:
                        continue
                    obj_class = fields[0]
                    if obj_class != self.class_name:
                        continue
                    bbox = [float(fields[4]), float(fields[5]), float(fields[6]), float(fields[7])]
                    bbox_width = bbox[2] - bbox[0]
                    bbox_height = bbox[3] - bbox[1]
                    if bbox_width >= self.min_patch_size and bbox_height >= self.min_patch_size:
                        self._patch_indices.append((img_file, label_file, bbox, fields))
        self.len = len(self._patch_indices)

    def __len__(self):
        return self.len

class KittiPatchIterator(KittiIterator):

    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)
        self.class_name = next((c for c in self.KITTI_CLASSES if c.lower() == class_name.lower()), None)
        assert self.class_name, f"Class '{class_name}' not found in KITTI classes."

    def __iter__(self):
        for img_file, label_file, bbox, fields in self._patch_indices:
            image_path = os.path.join(self.image_dir, img_file)
            image = Image.open(image_path).convert("RGB")
            x_min, y_min, x_max, y_max = map(int, bbox)
            if self.paint_bboxes:
                image = paint_bbox(image, [x_min, y_min, x_max, y_max], color=self.bbox_color, thickness=self.bbox_thickness)
            patch, patch_top_left  = crop(
                image,
                bbox=[x_min, y_min, x_max, y_max],
                pad_factor=self.patch_pad_factor,
                fill_color=self.patch_fill_color
            )
            # KITTI label fields: type, truncated, occluded, alpha, bbox (4), dimensions (3), location (3), rotation_y
            patch_data = PatchData(
                original_image_path=os.path.relpath(image_path, self.root),
                bbox_width=x_max - x_min,
                bbox_height=y_max - y_min,
                bbox_top_left=[x_min, y_min],
                patch_width=patch.width,
                patch_height=patch.height,
                patch_top_left=patch_top_left,
                location="Karlruhe",
                category=self.class_name,
                sub_categories=None,
                attributes=None,
                log=None,
                vehicle="Annieway",
                local_datetime=None,
                camera=None,
                weather=None,
                speed=None,
                acceleration=None,
                yaw_rate=None
            )
            yield patch, patch_data


class KittiImageIterator(KittiIterator):
    """Iterate over all KITTI images in the chosen split.

    Yields one downscaled full-image patch per image. Images are not
    duplicated. Bbox spans the entire (downscaled) image; instance-specific
    fields are None. Images without any labeled objects are also included.
    """

    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)

        if split in (DatasetSplit.TRAINVAL, DatasetSplit.TRAIN):
            dataset_root = os.path.join(root, "training")
        elif split == DatasetSplit.TEST:
            dataset_root = os.path.join(root, "testing")
        else:
            raise ValueError(f"Unsupported split '{split}' for KITTI dataset.")

        self.image_dir = os.path.join(dataset_root, "image_2")
        image_files = sorted(
            f for f in os.listdir(self.image_dir) if f.endswith(".png") or f.endswith(".jpg")
        )

        # Include all images, even if there are no labeled objects.
        self._image_files: list[str] = list(image_files)
        self.len = len(self._image_files)

    def __len__(self):
        return self.len

    def __iter__(self):
        for img_file in self._image_files:
            image_path = os.path.join(self.image_dir, img_file)
            image = Image.open(image_path).convert("RGB")
            patch = _downscale_full_image(image)
            bbox_width, bbox_height = patch.size
            patch_data = PatchData(
                original_image_path=os.path.relpath(image_path, self.root),
                bbox_width=bbox_width,
                bbox_height=bbox_height,
                bbox_top_left=[0, 0],
                location="Karlruhe",
                category=None,
                sub_categories=None,
                attributes=None,
                log=None,
                vehicle="Annieway",
                local_datetime=None,
                camera=None,
                weather=None,
                speed=None,
                acceleration=None,
                yaw_rate=None,
            )
            yield patch, patch_data


class WaymoIterator(PatchIterator):
    def __init__(self, root: str, dataset_name: str, class_name: str, split: DatasetSplit, min_patch_size: PositiveInt = 50, patch_pad_factor: float = 0, patch_fill_color: Tuple[PositiveInt] | None = None, paint_bboxes: bool = False, bbox_color: Tuple[PositiveInt] = (255, 0, 0), bbox_thickness: PositiveInt = 5):
        super().__init__(root, dataset_name, class_name, split, min_patch_size, patch_pad_factor, patch_fill_color, paint_bboxes, bbox_color, bbox_thickness)
        import waymo_open_dataset.v2 as wod
        self.wod = wod

        self.WAYMO_CLASSES = [cat.name.removeprefix("TYPE_").lower() for cat in wod.perception.box.BoxType]

        self.class_name = class_name.lower()

    @abstractmethod
    def _join_two_components(self, left, right, how='inner', group_left_by_common_keys=False, group_right_by_common_keys=False):
        pass

    def _join_components(self, components, how='inner', group_left_by_common_keys=False, group_right_by_common_keys=False):
        assert len(components) >= 2, "At least two components are required to join."
        result = components[0]
        for comp in components[1:]:
            result = self._join_two_components(result, comp, how=how, group_left_by_common_keys=group_left_by_common_keys, group_right_by_common_keys=group_right_by_common_keys)
        return result

    def _get_timezone_from_location(self, location: str) -> zoneinfo.ZoneInfo | None:
        if location == "location_phx":
            return zoneinfo.ZoneInfo("America/Phoenix")
        elif location == "location_sf":
            return zoneinfo.ZoneInfo("America/Los_Angeles")
        elif location == "location_other":
            return None
        else:
            raise ValueError(f"Unknown location '{location}' for timezone conversion.")

class WaymoPolarsIterator(WaymoIterator):
    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, dataset_name="waymo", class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)

        import polars as pl
        self.pl = pl

    def _join_two_components(self, left, right, how='inner', group_left_by_common_keys=False, group_right_by_common_keys=False):
        # ported https://github.com/waymo-research/waymo-open-dataset/blob/99a4cb3ff07e2fe06c2ce73da001f850f628e45a/src/waymo_open_dataset/v2/dataframe_utils.py#L55 to polars
        def _select_key_columns(df) -> set[str]:
            prefix = 'key.'
            return set([c for c in df.collect_schema().names() if c.startswith(prefix)])

        left_keys = _select_key_columns(left)
        right_keys = _select_key_columns(right)
        common_keys = left_keys.intersection(right_keys)

        def _group_by(src, keys):
            return src.group_by(keys).agg(self.pl.all())
        if group_left_by_common_keys and left_keys != common_keys:
            left = _group_by(left, common_keys)
        if group_right_by_common_keys and right_keys != common_keys:
            right = _group_by(right, common_keys)
        return left.join(right, on=common_keys, how=how)

    def read(self, tag: str, ids = None):
        """Creates a (lazy) Polars DataFrame for the component specified by its tag. Optionally filters by the ids given in the `ids` DataFrame."""
        parquet = self.pl.scan_parquet(f'{self.split_dir}/{tag}/*.parquet', low_memory=True).drop("index", strict=False) #, low_memory=True, cache=False
        if ids is not None:
            ids = ids.collect() if not isinstance(ids, self.pl.DataFrame) else ids
            for col in ids.collect_schema().names():
                parquet = parquet.filter(self.pl.col(col).is_in(ids[col].unique().implode()))
        return parquet


class WaymoPatchIterator(WaymoPolarsIterator):

    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)

        self.wod_class = next((c for c in self.wod.perception.box.BoxType if self.class_name in c.name.lower()), None)
        assert self.wod_class, f"Class '{class_name}' not found in Waymo classes. Available classes: {self.WAYMO_CLASSES}"

        assert split in (DatasetSplit.TRAIN, DatasetSplit.VAL), f"Waymo v2 only supports 'train' and 'val' splits."
        self.split_dir = os.path.join(root, "training" if split == DatasetSplit.TRAIN else "validation")


        camera_box_df = self.read('camera_box')
        stats_df = self.read('stats')
        # filter by class and min bbox size and group all available boxes per object, keep only the largest box per object (in case of multiple detections)
        keys = ['key.segment_context_name', 'key.camera_object_id']
        camera_box_df = (
            camera_box_df.filter((self.pl.col("[CameraBoxComponent].type") == self.wod_class)
                                 & (self.pl.col("[CameraBoxComponent].box.size.x") >= self.min_patch_size) & (self.pl.col("[CameraBoxComponent].box.size.y") >= self.min_patch_size))
            .group_by(keys)
            .agg(self.pl.all().get((self.pl.col("[CameraBoxComponent].box.size.x") * self.pl.col("[CameraBoxComponent].box.size.y")).arg_max()))
        )

        # join with camera images and metadata
        camera_box_df = self._join_components([camera_box_df, stats_df])
        self.len = camera_box_df.select(self.pl.len()).collect(engine="streaming").item() # lazy len: https://stackoverflow.com/questions/75523498/python-polars-how-to-get-the-row-count-of-a-lazyframe/75523731#75523731
        logging.info(f"Found {self.len} instances of class '{self.class_name}' in Waymo {split} split.")
        images = self.read('camera_image')
        merged_df = self._join_two_components(camera_box_df, images, how="inner")
        self.slice_iterator = merged_df.collect_batches(chunk_size=10, maintain_order=True, engine="streaming")


    def __len__(self):
        return self.len

    def __iter__(self):
        for slice in self.slice_iterator:
            for row in slice.iter_rows(named=True):
                camera_image = self.wod.CameraImageComponent.from_dict(row)
                camera_box = self.wod.CameraBoxComponent.from_dict(row)
                stats = self.wod.StatsComponent.from_dict(row)

                img = Image.open(BytesIO(camera_image.image)).convert("RGB")
                cx = camera_box.box.center.x
                cy = camera_box.box.center.y
                w = camera_box.box.size.x
                h = camera_box.box.size.y
                x_min = int(cx - w / 2)
                y_min = int(cy - h / 2)
                x_max = int(cx + w / 2)
                y_max = int(cy + h / 2)
                if self.paint_bboxes:
                    img = paint_bbox(img, [x_min, y_min, x_max, y_max], color=self.bbox_color, thickness=self.bbox_thickness)
                patch, patch_top_left  = crop(
                    img,
                    bbox=[x_min, y_min, x_max, y_max],
                    pad_factor=self.patch_pad_factor,
                    fill_color=self.patch_fill_color
                )
                bbox_width = x_max - x_min
                bbox_height = y_max - y_min
                bbox_top_left = [x_min, y_min]
                timezone = self._get_timezone_from_location(stats.location)
                patch_data = PatchData(
                    original_image_path=f"{camera_image.key.camera_name}:{camera_image.key.segment_context_name}:{camera_image.key.frame_timestamp_micros}",
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    bbox_top_left=bbox_top_left,
                    patch_width=patch.width,
                    patch_height=patch.height,
                    patch_top_left=patch_top_left,
                    location=stats.location,
                    category=self.class_name,
                    sub_categories=None,
                    attributes=None,
                    log=camera_image.key.segment_context_name,
                    vehicle=None,
                    local_datetime=None if timezone is None else datetime.datetime.fromtimestamp(camera_image.key.frame_timestamp_micros / 1e6, tz=timezone),
                    camera=self.wod.perception.camera_image.CameraName(camera_image.key.camera_name).name.lower(),
                    weather=stats.weather,
                    speed=(camera_image.velocity.linear_velocity.x**2 + camera_image.velocity.linear_velocity.y**2 + camera_image.velocity.linear_velocity.z**2)**0.5,
                    acceleration=None,
                    yaw_rate=camera_image.velocity.angular_velocity.z
                )
                yield patch, patch_data

class WaymoImageIterator(WaymoPolarsIterator):
    """Iterate over all Waymo images in the chosen split.

    Yields one downscaled full-image patch per image. Images are not
    duplicated. Bbox spans the entire (downscaled) image; instance-specific
    fields are None. Images without any labeled objects are also included.
    """

    def __init__(self, root: str, class_name: str, split: DatasetSplit, min_patch_size: int = 50, patch_pad_factor: float = 0.0, patch_fill_color: tuple[int, int, int]|None = None, paint_bboxes: bool = False, bbox_color: tuple[int, int, int] = (255, 0, 0), bbox_thickness: int = 5):
        super().__init__(root=root, class_name=class_name, split=split, min_patch_size=min_patch_size, patch_pad_factor=patch_pad_factor, patch_fill_color=patch_fill_color, paint_bboxes=paint_bboxes, bbox_color=bbox_color, bbox_thickness=bbox_thickness)


        assert split in (DatasetSplit.TRAIN, DatasetSplit.VAL), f"Waymo v2 only supports 'train' and 'val' splits."
        self.split_dir = os.path.join(root, "training" if split == DatasetSplit.TRAIN else "validation")

        self.stats_df = self.read('stats')
        self.len = self.stats_df.collect().height
        images = self.read('camera_image')
        merged_df = self._join_two_components(self.stats_df, images, how="inner")
        self.slice_iterator = merged_df.collect_batches(chunk_size=10, maintain_order=True, engine="streaming")

    def __len__(self):
        return self.len

    def __iter__(self):
        for slice in self.slice_iterator:
            for row in slice.iter_rows(named=True):
                camera_image = self.wod.CameraImageComponent.from_dict(row)
                stats = self.wod.StatsComponent.from_dict(row)

                img = Image.open(BytesIO(camera_image.image)).convert("RGB")
                patch = _downscale_full_image(img)
                bbox_width, bbox_height = patch.size
                timezone = self._get_timezone_from_location(stats.location)
                patch_data = PatchData(
                    original_image_path=f"{camera_image.key.camera_name}:{camera_image.key.segment_context_name}:{camera_image.key.frame_timestamp_micros}",
                    bbox_width=bbox_width,
                    bbox_height=bbox_height,
                    bbox_top_left=[0, 0],
                    location=stats.location,
                    category=None,
                    sub_categories=None,
                    attributes=None,
                    log=camera_image.key.segment_context_name,
                    vehicle=None,
                    local_datetime=None if timezone is None else datetime.datetime.fromtimestamp(camera_image.key.frame_timestamp_micros / 1e6, tz=timezone),
                    camera=self.wod.perception.camera_image.CameraName(camera_image.key.camera_name).name.lower(),
                    weather=stats.weather,
                    speed=(camera_image.velocity.linear_velocity.x**2 + camera_image.velocity.linear_velocity.y**2 + camera_image.velocity.linear_velocity.z**2)**0.5,
                    acceleration=None,
                    yaw_rate=camera_image.velocity.angular_velocity.z
                )
                yield patch, patch_data


class WaymoPatchFromJsonIterator(WaymoPolarsIterator):
    """Iterator to extract image patches from pre-existing dataset json files. Pad and paint bbox as specified in the json file."""

    def __init__(self, root: str, json_path: str):

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Dataset json file not found at {json_path}. Please run the patch extraction first to create the json file.")
        with open(json_path, 'r') as f:
            dataset_json = f.read()
        self.dataset = PatchDataset.model_validate_json(dataset_json)
        self.len = len(self.dataset.patches)
        super().__init__(root=root, class_name=self.dataset.class_name, split=self.dataset.split, min_patch_size=self.dataset.min_patch_size, patch_pad_factor=self.dataset.padding_factor, patch_fill_color=self.dataset.padding_color, paint_bboxes=self.dataset.bbox_color is not None, bbox_color=self.dataset.bbox_color if self.dataset.bbox_color is not None else (255, 0, 0), bbox_thickness=self.dataset.bbox_thickness if self.dataset.bbox_color is not None else 0)
        self.wod_class = next((c for c in self.wod.perception.box.BoxType if self.class_name in c.name.lower()), None)
        assert self.wod_class, f"Class '{self.class_name}' not found in Waymo classes. Available classes: {self.WAYMO_CLASSES}"

        assert self.dataset.split in (DatasetSplit.TRAIN, DatasetSplit.VAL), f"Waymo v2 only supports 'train' and 'val' splits."
        self.split_dir = os.path.join(root, "training" if self.dataset.split == DatasetSplit.TRAIN else "validation")

    def __len__(self):
        return self.len

    def _load_image(self, key_camera_name, key_segment_context_name, key_frame_timestamp_micros):
        # load image from Waymo tfrecord based on the key fields stored in the json file
        import waymo_open_dataset.v2 as wod

        # create a temporary Polars DataFrame with the key fields to filter the camera_image component
        key_df = self.pl.DataFrame({
            "key.camera_name": [int(key_camera_name)],
            "key.segment_context_name": [key_segment_context_name],
            "key.frame_timestamp_micros": [int(key_frame_timestamp_micros)]
        })
        images = self.read('camera_image', ids=key_df)
        image_row = images.collect().row(0, named=True)
        camera_image = wod.CameraImageComponent.from_dict(image_row)
        img = Image.open(BytesIO(camera_image.image)).convert("RGB")
        return img

    def __iter__(self):
        for patch_entry in self.dataset.patches:
            image = self._load_image(*patch_entry.patch_data.original_image_path.split(":"))
            x_min, y_min = patch_entry.patch_data.bbox_top_left
            x_max = x_min + patch_entry.patch_data.bbox_width
            y_max = y_min + patch_entry.patch_data.bbox_height
            if self.paint_bboxes:
                image = paint_bbox(image, [x_min, y_min, x_max, y_max], color=self.bbox_color, thickness=self.bbox_thickness)
            patch, patch_top_left  = crop(
                image,
                bbox=[x_min, y_min, x_max, y_max],
                pad_factor=self.patch_pad_factor,
                fill_color=self.patch_fill_color
            )
            assert patch_top_left == patch_entry.patch_data.patch_top_left, f"Patch top left corner after cropping does not match the one stored in the json file for patch {patch_entry.patchpath}. This indicates an inconsistency between the stored patch data and the actual image. Please check the json file and the image at {image_path} for this patch."
            yield patch, patch_entry.patch_data, patch_entry.patchpath


def extract_patches_from_json(json_path: str, dataset_root: str, output_root: str, overwrite_existing_patches: bool = False):
    """Extract image patches from a pre-existing dataset json file and save them to the same location as the json file."""
    dataset = load_patch_dataset_from_json(json_path)
    extract_patches_from_patch_dataset(dataset, json_path=json_path, dataset_root=dataset_root, output_root=output_root, overwrite_existing_patches=overwrite_existing_patches)

def load_patch_dataset_from_json(json_path: str) -> PatchDataset:
    """Load a PatchDataset object from a json file."""
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset json file not found at {json_path}.")
    with open(json_path, 'r') as f:
        dataset_json = f.read()
        return PatchDataset.model_validate_json(dataset_json)

def extract_patches_from_patch_dataset(dataset: PatchDataset, json_path: str, dataset_root: str, output_root: str, overwrite_existing_patches: bool = False):
    if dataset.dataset_name == "waymo":
        iterator = WaymoPatchFromJsonIterator(root=dataset_root, json_path=json_path)
    else:
        raise NotImplementedError(f"Patch extraction from json files is only implemented for Waymo dataset for now. Dataset '{dataset.dataset_name}' is not supported.")
    for patch, patch_data, patch_path in tqdm(iterator, desc=f"Extracting patches from json file {json_path}", leave=True):
        patch_path = os.path.join(output_root, patch_path)
        if not Path(patch_path).parent.is_dir():
            if Path(patch_path).parent.exists():
                raise FileExistsError(f"Expected {Path(patch_path).parent} to be a directory, but found a file with the same name. Please remove or rename the file and try again.")
            os.makedirs(Path(patch_path).parent)
        if overwrite_existing_patches or not os.path.exists(patch_path):
            patch.save(patch_path)
    logging.info(f"Extracted and saved all patches from json file {json_path}.")

def extract_and_save_patches(iterator: PatchIterator, output_dir: str):
    """
    Extracts patches from the dataset and saves them to the specified output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset_name = iterator.dataset_name
    split = iterator.split.value
    class_name = iterator.class_name
    if class_name != "image":
        padding_factor = iterator.patch_pad_factor
        padding_color = iterator.patch_fill_color
        bbox_painted = iterator.paint_bboxes
        bbox_color = iterator.bbox_color
        bbox_thickness = iterator.bbox_thickness
        patches = []
        padding_color_str = f'_c{padding_color[0]}-{padding_color[1]}-{padding_color[2]}' if padding_color is not None else '_clip'
        padding_suffix = f'_pad{str(padding_factor).replace(".", "-") + padding_color_str if padding_factor else ""}' if padding_factor > 0 else ''
        bbox_suffix = f'_bbox{str(bbox_color[0])}-{str(bbox_color[1])}-{str(bbox_color[2])}_th{bbox_thickness}' if bbox_painted else ''
        padding_suffix += bbox_suffix
    else:
        padding_factor = 0.
        padding_color = None
        bbox_painted = False
        bbox_color = (0.,0.,0.)
        bbox_thickness = 0
        patches = []
        padding_suffix = ""

    json_dir = os.path.join(output_dir, dataset_name, split, class_name)
    img_dir = os.path.join(json_dir, f"images{padding_suffix}")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    for i, (patch, patch_data) in enumerate(tqdm(iterator, desc=f"Extracting patches for {class_name} instances from {dataset_name} ({split})")):
        patch_filename = f"{dataset_name}_{split}_{class_name}_{i:06d}.jpg"
        patch_path = os.path.join(img_dir, patch_filename)
        patch.save(patch_path)

        patches.append(PatchEntry(patchpath=patch_path, patch_data=patch_data))

    dataset = PatchDataset(dataset_name=dataset_name, class_name=class_name, split=iterator.split, patches=patches, min_patch_size=iterator.min_patch_size, padding_factor=padding_factor, padding_color=padding_color, bbox_color=bbox_color if paint_bbox else None, bbox_thickness=bbox_thickness if paint_bbox else 0)

    logging.info(f"Extracted {len(patches)} patches for {dataset_name} ({split}) and saved to {output_dir}")

    json_filename = f"{dataset_name}_{split}_{class_name}_patches{padding_suffix}.json"
    json_path = os.path.join(json_dir, json_filename)
    logging.info(f"Saving dataset json to {json_path}...")
    with open(json_path, 'w') as f:
        f.write(dataset.model_dump_json(indent=4))
    logging.info("Dataset saved successfully.")
    return dataset

def main_recreate(args):
    if not os.path.exists(args.path):
        raise FileNotFoundError(f"Path '{args.path}' does not exist.")
    if args.output_root is None:
        output_root = data_dir
    else:
        output_root = args.output_root
    if os.path.isfile(args.path):
        extract_patches_from_json(args.path, dataset_root=args.root, output_root=output_root, overwrite_existing_patches=args.overwrite)
    elif os.path.isdir(args.path):
        json_files = list(Path(args.path).rglob(args.json_pattern))

        if not any(json_files):
            raise FileNotFoundError(f"No json files found in directory '{args.path}'. Please provide a valid path to a dataset json file or a directory containing dataset json files.")
        for json_file in tqdm(json_files, desc="Processing json files"):
            dataset = load_patch_dataset_from_json(json_file)
            if args.include_non_default_padding_variants:
                extract_patches_from_patch_dataset(dataset, json_path=str(json_file), dataset_root=args.root, output_root=output_root, overwrite_existing_patches=args.overwrite)
            else:
                match dataset:
                    case PatchDataset(padding_factor=0.5, bbox_color=(255, 0, 0), bbox_thickness=5, padding_color=None):
                        extract_patches_from_patch_dataset(dataset, json_path=str(json_file), dataset_root=args.root, output_root=output_root, overwrite_existing_patches=args.overwrite)
                    case PatchDataset():
                        logging.info(f"Skipping json file {json_file} with non-default padding or bbox parameters (padding_factor={dataset.padding_factor}, bbox_color={dataset.bbox_color}, bbox_thickness={dataset.bbox_thickness}, padding_color={dataset.padding_color}). Use the '--include_non_default_padding_variants' flag to include these variants in the patch extraction.")
                    case _:
                        assert False, f"dataset should be an instance of PatchDataset, but got {type(dataset)}"
    else:
        raise ValueError(f"Path '{args.path}' is neither a file nor a directory.")


def main_create(args):

    print(f"Extracting patches for the datasets {args.dataset} with roots {args.root} and class names {args.class_name}.")

    if len(args.dataset) != len(args.root):
        raise ValueError("The number of datasets and root directories must be the same.")
    if len(args.class_name) != len(args.dataset) and len(args.class_name) != 1:
        raise ValueError("The number of class names must be either 1 or equal to the number of datasets.")
    for i, dataset in enumerate(args.dataset):
        root = args.root[i]
        class_name = args.class_name[i] if len(args.class_name) > 1 else args.class_name[0]
        print(f"Processing dataset: {dataset}, root: {root}, class_name: {class_name}")
        if class_name.lower() == "all":
            if dataset == "nuimages":
                from nuimages import NuImages
                nuimages = NuImages(version=f"v1.0-{args.split}", dataroot=os.path.expanduser(root), lazy=False, verbose=True)
                class_names = list(set(cat["name"].split('.', 1)[0] for cat in nuimages.category)) # get unique super categories
            elif dataset == "kitti":
                class_names = KittiPatchIterator.KITTI_CLASSES
            elif dataset == "waymo":
                import waymo_open_dataset.v2 as wod
                class_names = [cat.name.removeprefix("TYPE_").lower() for cat in wod.perception.box.BoxType]
            else:
                raise ValueError(f"Unsupported dataset: {dataset}")
            logging.info(f"Extracting patches for all classes in {dataset}: {class_names}")
            for cls in class_names:
                print(f"Extracting patches for class {cls} of dataset {dataset}...")
                args_single = argparse.Namespace(
                    dataset=dataset,
                    root=root,
                    class_name=cls,
                    split=args.split,
                    min_patch_size=args.min_patch_size,
                    output_dir=args.output_dir,
                    patch_pad_factor=args.patch_pad_factor,
                    patch_fill_color=args.patch_fill_color,
                    paint_bboxes=args.paint_bboxes,
                    bbox_thickness=args.bbox_thickness
                )
                main_create_single(args_single)
        else:
            args_single = argparse.Namespace(
                dataset=dataset,
                root=root,
                class_name=class_name,
                split=args.split,
                min_patch_size=args.min_patch_size,
                output_dir=args.output_dir,
                patch_pad_factor=args.patch_pad_factor,
                patch_fill_color=args.patch_fill_color,
                paint_bboxes=args.paint_bboxes,
                bbox_thickness=args.bbox_thickness
            )
            main_create_single(args_single)

def main_create_single(args):
    args.root = os.path.expanduser(args.root)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    if args.dataset == "nuimages":
        if args.class_name == "image":
            iterator = NuImagesImageIterator(
                root=args.root,
                class_name="image",
                split=DatasetSplit(args.split),
                min_patch_size=args.min_patch_size,
                patch_pad_factor=0.0,
                patch_fill_color=None,
                paint_bboxes=False,
                bbox_thickness=args.bbox_thickness
            )
        else:
            iterator = NuImagesPatchIterator(
                root=args.root,
                class_name=args.class_name,
                split=DatasetSplit(args.split),
                min_patch_size=args.min_patch_size,
                patch_pad_factor=args.patch_pad_factor,
                patch_fill_color=tuple(map(int, args.patch_fill_color.split(','))) if args.patch_fill_color else None,
                paint_bboxes=args.paint_bboxes,
                bbox_thickness=args.bbox_thickness
            )
    elif args.dataset == "kitti":
        if args.class_name == "image":
            iterator = KittiImageIterator(
                root=args.root,
                class_name="image",
                split=DatasetSplit(args.split),
                min_patch_size=args.min_patch_size,
                patch_pad_factor=0.0,
                patch_fill_color=None,
                paint_bboxes=False,
                bbox_thickness=args.bbox_thickness
            )
        else:
            iterator = KittiPatchIterator(
                root=args.root,
                class_name=args.class_name,
                split=DatasetSplit(args.split),
                min_patch_size=args.min_patch_size,
                patch_pad_factor=args.patch_pad_factor,
                patch_fill_color=tuple(map(int, args.patch_fill_color.split(','))) if args.patch_fill_color else None,
                paint_bboxes=args.paint_bboxes,
                bbox_thickness=args.bbox_thickness
        )
    elif args.dataset == "waymo":
        if args.class_name == "image":
            iterator = WaymoImageIterator(
                root=args.root,
                class_name="image",
                split=DatasetSplit(args.split),
                min_patch_size=args.min_patch_size,
                patch_pad_factor=0.0,
                patch_fill_color=None,
                paint_bboxes=False,
                bbox_thickness=args.bbox_thickness
            )
        else:
            iterator = WaymoPatchIterator(
                root=args.root,
                class_name=args.class_name,
                split=DatasetSplit(args.split),
                min_patch_size=args.min_patch_size,
                patch_pad_factor=args.patch_pad_factor,
                patch_fill_color=tuple(map(int, args.patch_fill_color.split(','))) if args.patch_fill_color else None,
                paint_bboxes=args.paint_bboxes,
                bbox_thickness=args.bbox_thickness
            )
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    logging.info(f"Extracting patches for {args.class_name} in {args.split} split from {args.root}")
    dataset = extract_and_save_patches(iterator, args.output_dir)


def main():
    parser = argparse.ArgumentParser(description="Extract image patches from a dataset.")
    subparsers = parser.add_subparsers(required=True)

    create_parser = subparsers.add_parser("create", help="Create a new dataset by extracting patches from the original dataset based on the specified parameters.")

    create_parser.add_argument("--dataset", type=str.lower, required=True,  nargs='+', default=[], choices=["nuimages", "kitti", "waymo"], help="Dataset to extract patches from.")
    create_parser.add_argument("--root", type=str, required=True, nargs='+', default=[], help="Root directory of the dataset.")
    create_parser.add_argument("--class_name", type=str, required=True,  nargs='+', default=[], help="Class name to extract patches for. Special value 'all' to extract patches for all classes.")
    create_parser.add_argument("--split", type=str, choices=[s.value for s in DatasetSplit], required=True, help="Dataset split to extract patches from.")
    create_parser.add_argument("--min_patch_size", type=int, default=50, help="Minimum patch size (in pixels) to consider.")
    create_parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the extracted patches.")
    create_parser.add_argument("--patch_pad_factor", type=float, default=0.0, help="Padding factor to apply to each side of the bounding box when extracting patches.")
    create_parser.add_argument("--patch_fill_color", type=str, default=None, help="Fill color (format: \"R,G,B\") to use for padding areas outside the image. If not set, patches are clipped to image boundaries.")
    create_parser.add_argument("--paint_bboxes", action="store_true", help="If set, paint bounding boxes on the original image before extracting patches.")
    create_parser.add_argument("--bbox_thickness", type=int, default=5, help="Thickness of the bounding box to paint if --paint_bboxes is set.")
    create_parser.set_defaults(func=main_create)

    recreate_parser = subparsers.add_parser("recreate", help="Recreate image patches from a pre-existing dataset json file. Use this to recreate the benchmark dataset from the downloaded json files and the official download of the Waymo dataset.")

    recreate_parser.add_argument("--path", type=str, required=True, help="Path to either a dataset json file to extract patches from or a directory containing multiple dataset json files to extract patches from all of them. If a directory is provided, the script will recursively scan the directory for json files and extract patches from them.")
    recreate_parser.add_argument("--root", type=str, required=True, help="Root directory of the dataset.")
    recreate_parser.add_argument("--output_root", type=str, required=False, help="Root directory to save the extracted patches. If not set, patches will be saved to the ad_data_dir specified in data_dir.json or the default ad_data_dir if not specified in data_dir.json (data/ad-datasets).")
    recreate_parser.add_argument("--overwrite", action="store_true", default=False, help="If set, overwrite existing patch images. Use with caution, as this will overwrite any existing patches at the same location without warning.")
    recreate_parser.add_argument("--json-pattern", type=str, default=r"*.json", help="Glob pattern to match json files when a directory is provided as the path. Only json files with names matching the pattern will be processed. Default is '*.json', which matches all json files.")
    recreate_parser.add_argument("--include-non-default-padding-variants", action="store_true", help="If set, process all json files that match the regex. If not set, only process json files with the default padding variant (padding factor 0.5, not centered, red (255,0,0) bbox with thickness 5). Extracting all padding variants will require significantly more storage space and time.")
    recreate_parser.set_defaults(func=main_recreate)

    args = parser.parse_args()
    args.func(args)
    print("Patch extraction completed successfully.")

if __name__ == "__main__":
    main()
