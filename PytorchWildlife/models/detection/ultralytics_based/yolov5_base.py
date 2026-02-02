# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the AGPL License.

""" YoloV5 base detector class. """

# Importing basic libraries

import os
import numpy as np
from tqdm import tqdm
from PIL import Image
import supervision as sv

import torch
from torch.utils.data import DataLoader
from torch.hub import load_state_dict_from_url

from yolov5.utils.general import non_max_suppression, scale_boxes
from ..base_detector import BaseDetector
from ....data import transforms as pw_trans
from ....data import datasets as pw_data


class YOLOV5Base(BaseDetector):
    """
    Base detector class for YOLO V5. This class provides utility methods for
    loading the model, generating results, and performing single and batch image detections.
    """
    def __init__(self, weights=None, device="cpu", url=None, transform=None):
        """
        Initialize the YOLO V5 detector.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
            transform (callable, optional):
                Optional transform to be applied on the image. Defaults to None.
        """
        self.transform = transform
        super(YOLOV5Base, self).__init__(weights=weights, device=device, url=url)
        self._load_model(weights, device, url)

    def _load_model(self, weights=None, device="cpu", url=None):
        """
        Load the YOLO V5 model weights.
        
        Args:
            weights (str, optional): 
                Path to the model weights. Defaults to None.
            device (str, optional): 
                Device for model inference. Defaults to "cpu".
            url (str, optional): 
                URL to fetch the model weights. Defaults to None.
        Raises:
            Exception: If weights are not provided.
        """
        if weights:
            checkpoint = torch.load(weights, map_location=torch.device(device))
        elif url:
            checkpoint = load_state_dict_from_url(url, map_location=torch.device(self.device))
        else:
            raise Exception("Need weights for inference.")
        self.model = checkpoint["model"].float().fuse().eval().to(self.device)
        
        if not self.transform:
            self.transform = pw_trans.MegaDetector_v5_Transform(target_size=self.IMAGE_SIZE,
                                                                stride=self.STRIDE)

    def results_generation(self, preds, img_id, id_strip=None) -> dict:
        """
        Generate results for detection based on model predictions.
        
        Args:
            preds (numpy.ndarray): 
                Model predictions.
            img_id (str): 
                Image identifier.
            id_strip (str, optional): 
                Strip specific characters from img_id. Defaults to None.

        Returns:
            dict: Dictionary containing image ID, detections, and labels.
        """
        img_id = str(img_id)
        if id_strip:
            try:
                img_id = os.path.relpath(img_id, start=id_strip)
            except Exception:
                prefix = id_strip + os.sep
                if img_id.startswith(prefix):
                    img_id = img_id[len(prefix):]

        results = {"img_id": img_id}
        results["detections"] = sv.Detections(
            xyxy=preds[:, :4],
            confidence=preds[:, 4],
            class_id=preds[:, 5].astype(int)
        )
        results["labels"] = [
            f"{self.CLASS_NAMES[class_id]} {confidence:0.2f}"
            for confidence, class_id in zip(results["detections"].confidence, results["detections"].class_id)
        ]
        return results

    def single_image_detection(self, img, img_path=None, det_conf_thres=0.2, id_strip=None) -> dict:
        """
        Perform detection on a single image.
        
        Args:
            img (str or ndarray): 
                Image path or ndarray of images.
            img_path (str, optional): 
                Image path or identifier.
            det_conf_thres (float, optional): 
                Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): 
                Characters to strip from img_id. Defaults to None.

        Returns:
            dict: Detection results.
        """
        if type(img) == str:
            if img_path is None:
                img_path = img
            img = np.array(Image.open(img_path).convert("RGB"))
        img_size = img.shape
        img = self.transform(img)

        if img_size is None:
            img_size = img.permute((1, 2, 0)).shape # We need hwc instead of chw for coord scaling
        preds = self.model(img.unsqueeze(0).to(self.device))[0]
        preds = torch.cat(non_max_suppression(prediction=preds, conf_thres=det_conf_thres), axis=0).cpu().numpy()
        # preds[:, :4] = scale_coords([self.IMAGE_SIZE] * 2, preds[:, :4], img_size).round()
        preds[:, :4] = scale_boxes([self.IMAGE_SIZE] * 2, preds[:, :4], img_size).round()
        res = self.results_generation(preds, img_path, id_strip)

        normalized_coords = [[x1 / img_size[1], y1 / img_size[0], x2 / img_size[1], y2 / img_size[0]] for x1, y1, x2, y2 in preds[:, :4]]
        res["normalized_coords"] = normalized_coords

        return res

    def batch_image_detection(
        self,
        data_path,
        batch_size: int = 16,
        det_conf_thres: float = 0.2,
        id_strip: str = None,
        show_paths: bool = False,
        path_log_every: int = 25,
        path_log_mode: str = "tqdm",
        num_workers: int = 0,
        prefetch_factor: int | None = None,
        persistent_workers: bool = False,
        pin_memory: bool = True,
        decoder: str = "pil",
        corrupt_log_path: str | None = None,
    ) -> list[dict]:
        """
        Perform detection on a batch of images.

        Args:
            data_path (str): Path containing all images for inference.
            batch_size (int, optional): Batch size for inference. Defaults to 16.
            det_conf_thres (float, optional): Confidence threshold for predictions. Defaults to 0.2.
            id_strip (str, optional): Characters to strip from img_id. Defaults to None.
            show_paths (bool, optional): If True, update the progress bar with the current image path.
            path_log_every (int, optional): If show_paths is True, also print the current path every N images.
            path_log_mode (str, optional): 'tqdm' (postfix) or 'line' (single-line overwrite).
            num_workers (int, optional): DataLoader worker processes for decode.
            prefetch_factor (int, optional): Prefetch batches per worker (requires num_workers > 0).
            persistent_workers (bool, optional): Keep workers alive between batches.
            pin_memory (bool, optional): Pin memory for faster H2D copies.
            decoder (str, optional): 'pil', 'opencv', or 'torchvision'.
            corrupt_log_path (str, optional): Path to log corrupt images.

        Returns:
            list[dict]: List of detection results for all images.
        """

        dataset = pw_data.DetectionImageFolder(
            data_path,
            transform=self.transform,
            decoder=decoder,
            corrupt_log_path=corrupt_log_path,
        )

        # Creating a DataLoader for batching and parallel processing of the images
        use_workers = max(0, int(num_workers or 0))
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=bool(pin_memory),
            num_workers=use_workers,
            drop_last=False,
            prefetch_factor=(prefetch_factor if use_workers > 0 else None),
            persistent_workers=(bool(persistent_workers) if use_workers > 0 else False),
            collate_fn=pw_data.collate_skip_none,
        )

        import sys
        import time

        results = []
        total_imgs = len(dataset)
        log_every = max(1, int(path_log_every))
        mode = (path_log_mode or "tqdm").lower()
        if mode == "inline":
            mode = "line"

        use_line = show_paths and mode == "line"
        if use_line:
            start_time = time.time()
            done = 0
            bar_len = 12

            for batch_index, batch in enumerate(loader):
                if batch is None:
                    continue
                imgs, paths, sizes = batch
                imgs = imgs.to(self.device)
                predictions = self.model(imgs)[0].detach().cpu()
                predictions = non_max_suppression(predictions, conf_thres=det_conf_thres)

                batch_results = []
                for i, pred in enumerate(predictions):
                    path = paths[i]
                    done += 1
                    if done % log_every == 0 or done == total_imgs:
                        elapsed = time.time() - start_time
                        rate = done / elapsed if elapsed > 0 else 0.0
                        remaining = (total_imgs - done) / rate if rate > 0 else 0.0
                        filled = int(bar_len * done / total_imgs) if total_imgs > 0 else 0
                        bar = "█" * filled + " " * (bar_len - filled)
                        pct = (done / total_imgs * 100) if total_imgs > 0 else 0
                        sys.stdout.write(
                            f"\rDetecting: {pct:3.0f}%|{bar}| {done}/{total_imgs} "
                            f"[{elapsed:0.0f}s<{remaining:0.0f}s, {rate:.2f}img/s, {path}]"
                        )
                        sys.stdout.flush()

                    if pred.size(0) == 0:
                        continue
                    pred = pred.numpy()
                    size = sizes[i].numpy()
                    original_coords = pred[:, :4].copy()
                    # pred[:, :4] = scale_coords([self.IMAGE_SIZE] * 2, pred[:, :4], size).round()
                    pred[:, :4] = scale_boxes([self.IMAGE_SIZE] * 2, pred[:, :4], size).round()
                    # Normalize the coordinates for timelapse compatibility
                    normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in pred[:, :4]]
                    res = self.results_generation(pred, path, id_strip)
                    res["normalized_coords"] = normalized_coords
                    batch_results.append(res)
                results.extend(batch_results)

            sys.stdout.write("\n")
            sys.stdout.flush()
            return results

        bar_format = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
        with tqdm(
            total=total_imgs,
            desc="Detecting",
            unit="img",
            dynamic_ncols=True,
            mininterval=0.2,
            smoothing=0.1,
            bar_format=bar_format,
        ) as pbar:
            for batch_index, batch in enumerate(loader):
                if batch is None:
                    continue
                imgs, paths, sizes = batch
                imgs = imgs.to(self.device)
                predictions = self.model(imgs)[0].detach().cpu()
                predictions = non_max_suppression(predictions, conf_thres=det_conf_thres)

                batch_results = []
                for i, pred in enumerate(predictions):
                    path = paths[i]
                    if show_paths and (pbar.n % log_every) == 0 and hasattr(pbar, "set_postfix_str"):
                        pbar.set_postfix_str(str(path), refresh=True)
                    if pred.size(0) == 0:
                        pbar.update(1)
                        continue
                    pred = pred.numpy()
                    size = sizes[i].numpy()
                    original_coords = pred[:, :4].copy()
                    # pred[:, :4] = scale_coords([self.IMAGE_SIZE] * 2, pred[:, :4], size).round()
                    pred[:, :4] = scale_boxes([self.IMAGE_SIZE] * 2, pred[:, :4], size).round()
                    # Normalize the coordinates for timelapse compatibility
                    normalized_coords = [[x1 / size[1], y1 / size[0], x2 / size[1], y2 / size[0]] for x1, y1, x2, y2 in pred[:, :4]]
                    res = self.results_generation(pred, path, id_strip)
                    res["normalized_coords"] = normalized_coords
                    batch_results.append(res)
                    pbar.update(1)
                results.extend(batch_results)
        return results
