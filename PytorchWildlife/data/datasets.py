# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import time
from glob import glob
from PIL import Image, ImageFile, UnidentifiedImageError
import numpy as np
import supervision as sv
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

# To handle truncated images during loading
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Making the DetectionImageFolder class available for import from this module
__all__ = [
    "DetectionImageFolder",
    "collate_skip_none",
    ]

# Define the allowed image extensions  
IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")  
  
def has_file_allowed_extension(filename: str, extensions: tuple) -> bool:  
    """Checks if a file is an allowed extension."""  
    return filename.lower().endswith(extensions if isinstance(extensions, str) else tuple(extensions))
  
def is_image_file(filename: str) -> bool:  
    """Checks if a file is an allowed image extension."""  
    return has_file_allowed_extension(filename, IMG_EXTENSIONS) 


def collate_skip_none(batch):
    """Collate function that drops None samples (e.g., corrupt images)."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    return default_collate(batch)

class ImageFolder(Dataset):
    """
    A PyTorch Dataset for loading images from a specified directory.
    Each item in the dataset is a tuple containing the image data, 
    the image's path, and the original size of the image.
    """

    def __init__(self, image_dir, transform=None):
        """
        Initializes the dataset.

        Parameters:
            image_dir (str): Path to the directory containing the images.
            transform (callable, optional): Optional transform to be applied on the image.
        """
        super(ImageFolder, self).__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.images = [os.path.join(dp, f) for dp, dn, filenames in os.walk(image_dir) for f in filenames if is_image_file(f)] # dp: directory path, dn: directory name, f: filename

    def __getitem__(self, idx) -> tuple:
        """
        Retrieves an image from the dataset.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Contains the image data, the image's path, and its original size.
        """
        pass
    
    def __len__(self) -> int:
        """
        Returns the total number of images in the dataset.

        Returns:
            int: Total number of images.
        """
        return len(self.images)

class ClassificationImageFolder(ImageFolder):
    """
    A PyTorch Dataset for loading images from a specified directory.
    Each item in the dataset is a tuple containing the image data, 
    the image's path, and the original size of the image.
    """

    def __init__(self, image_dir, transform=None):
        """
        Initializes the dataset.

        Parameters:
            image_dir (str): Path to the directory containing the images.
            transform (callable, optional): Optional transform to be applied on the image.
        """
        super(ClassificationImageFolder, self).__init__(image_dir, transform)

    def __getitem__(self, idx) -> tuple:
        """
        Retrieves an image from the dataset.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Contains the image data, the image's path, and its original size.
        """
        # Get image filename and path
        img_path = self.images[idx]

        # Load and convert image to RGB
        img = Image.open(img_path).convert("RGB")
        
        # Apply transformation if specified
        if self.transform:
            img = self.transform(img)

        return img, img_path


class DetectionImageFolder(ImageFolder):
    """
    A PyTorch Dataset for loading images from a specified directory.
    Each item in the dataset is a tuple containing the image data, 
    the image's path, and the original size of the image.
    """

    def __init__(self, image_dir, transform=None, decoder: str = "pil", corrupt_log_path: str | None = None):
        """
        Initializes the dataset.

        Parameters:
            image_dir (str): Path to the directory containing the images.
            transform (callable, optional): Optional transform to be applied on the image.
        """
        super(DetectionImageFolder, self).__init__(image_dir, transform)
        self.decoder = (decoder or "pil").lower()
        self.corrupt_log_path = corrupt_log_path

    def _log_corrupt(self, img_path: str, exc: BaseException) -> None:
        if not self.corrupt_log_path:
            return
        try:
            log_dir = os.path.dirname(self.corrupt_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
            ts = time.strftime("%Y-%m-%d %H:%M:%S")
            with open(self.corrupt_log_path, "a", encoding="utf-8") as f:
                f.write(f"[{ts}] {img_path} | {type(exc).__name__}: {exc}\n")
        except Exception:
            pass

    def __getitem__(self, idx) -> tuple:
        """
        Retrieves an image from the dataset.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Contains the image data, the image's path, and its original size.
        """
        # Get image filename and path
        img_path = self.images[idx]

        # Load and convert image to RGB
        img = None
        try:
            if self.decoder == "opencv":
                try:
                    import cv2
                    bgr = cv2.imread(img_path, cv2.IMREAD_COLOR)
                    if bgr is None:
                        raise RuntimeError(f"cv2.imread failed for {img_path}")
                    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                except Exception:
                    img = Image.open(img_path).convert("RGB")
            elif self.decoder == "torchvision":
                try:
                    from torchvision.io import read_image
                    t = read_image(img_path)  # CxHxW, uint8
                    img = t.permute(1, 2, 0).numpy()
                except Exception:
                    img = Image.open(img_path).convert("RGB")
            else:
                img = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError, RuntimeError) as exc:
            self._log_corrupt(img_path, exc)
            return None
        except Exception as exc:
            self._log_corrupt(img_path, exc)
            return None

        if isinstance(img, Image.Image):
            img_size_ori = img.size[::-1]
        else:
            img_size_ori = (img.shape[0], img.shape[1])
        
        # Apply transformation if specified
        if self.transform:
            img = self.transform(img)

        return img, img_path, torch.tensor(img_size_ori)
    

# TODO: Under development for efficiency improvement
class DetectionCrops(Dataset):

    def __init__(self, detection_results, transform=None, path_head=None, animal_cls_id=0):

        self.detection_results = detection_results
        self.transform = transform
        self.path_head = path_head
        self.animal_cls_id = animal_cls_id # This determines which detection class id represents animals.
        self.img_ids = []
        self.xyxys = []

        self.load_detection_results()

    def load_detection_results(self):
        for det in self.detection_results:
            for xyxy, det_id in zip(det["detections"].xyxy, det["detections"].class_id):
                # Only run recognition on animal detections
                if det_id == self.animal_cls_id:
                    self.img_ids.append(det["img_id"])
                    self.xyxys.append(xyxy)

    def __getitem__(self, idx) -> tuple:
        """
        Retrieves an image from the dataset.

        Parameters:
            idx (int): Index of the image to retrieve.

        Returns:
            tuple: Contains the image data and the image's path.
        """

        # Get image path and corresponding bbox xyxy for cropping
        img_id = self.img_ids[idx]
        xyxy = self.xyxys[idx]

        img_path = os.path.join(self.path_head, img_id) if self.path_head else img_id
        
        # Load and crop image with supervision
        img = sv.crop_image(np.array(Image.open(img_path).convert("RGB")),
                            xyxy=xyxy)
        
        # Apply transformation if specified
        if self.transform:
            img = self.transform(Image.fromarray(img))

        return img, img_path

    def __len__(self) -> int:
        return len(self.img_ids)