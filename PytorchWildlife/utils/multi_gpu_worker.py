"""Multi-GPU multiprocessing worker for notebook use."""

from __future__ import annotations

import os
import sys
import time
import inspect
import re
import threading
from typing import Any

import torch

from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils


def _count_images(folder_path: str) -> int:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
    count = 0
    for root, _, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                count += 1
    return count


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg or ("cuda" in msg and "memory" in msg)


def _log(msg: str, prefix: str | None = None) -> None:
    if prefix:
        msg = f"[{prefix}] {msg}"
    print(msg, flush=True)


class _PathTracker:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_path: str | None = None

    def update_from_text(self, text: str) -> None:
        matches = re.findall(
            r"(/[^\s]+\.(?:jpg|jpeg|png|bmp|tif|tiff|gif|webp))",
            text,
            flags=re.IGNORECASE,
        )
        if matches:
            path = matches[-1].rstrip("])")
            with self._lock:
                self._last_path = path

    def get_last_path(self) -> str | None:
        with self._lock:
            return self._last_path


class _TimestampedWriter:
    def __init__(self, stream):
        self._stream = stream

    def write(self, s: str) -> int:
        if not s:
            return 0
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        lines = s.splitlines(True)
        out = "".join(
            f"[{ts}] {line}" if line.strip() else line for line in lines
        )
        return self._stream.write(out)

    def flush(self) -> None:
        self._stream.flush()


class _TrackingWriter(_TimestampedWriter):
    def __init__(self, stream, tracker: _PathTracker):
        super().__init__(stream)
        self._tracker = tracker

    def write(self, s: str) -> int:
        if s:
            self._tracker.update_from_text(s)
        return super().write(s)


def _heartbeat_loop(stop_event: threading.Event, tracker: _PathTracker, interval: int, prefix: str) -> None:
    while not stop_event.wait(interval):
        current = tracker.get_last_path() or "(unknown)"
        _log(f"heartbeat current_file={current}", prefix=prefix)


def process_folder_on_gpu(folder_path: str, gpu_id: int, cfg: dict[str, Any], queue) -> None:
    """Process a single folder on a specific GPU."""
    try:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        if cfg["model_family"] == "MegaDetectorV6":
            detection_model_local = pw_detection.MegaDetectorV6(
                device=device, pretrained=True, version=cfg["model_version"]
            )
        else:
            detection_model_local = pw_detection.MegaDetectorV5(
                device=device, pretrained=True, version=cfg["model_version"]
            )

        folder_leaf = os.path.basename(os.path.normpath(folder_path))
        json_file = os.path.join(cfg["output_root"], f"detection_results__{folder_leaf}.json")

        if cfg["resume_skip"] and os.path.isfile(json_file):
            queue.put({"status": "skipped_existing_json", "folder": folder_path, "json_file": json_file})
            return

        img_count = _count_images(folder_path)
        if cfg["skip_empty"] and img_count == 0:
            queue.put({"status": "skipped_empty", "folder": folder_path, "images": 0})
            return

        if cfg["dry_run"]:
            queue.put({"status": "dry_run", "folder": folder_path, "images": img_count})
            return

        def _run_batch_detection_local(bs: int):
            cur_bs = bs
            while True:
                try:
                    if cfg.get("show_paths", False) and "show_paths" in inspect.signature(
                        detection_model_local.batch_image_detection
                    ).parameters:
                        return detection_model_local.batch_image_detection(
                            folder_path,
                            batch_size=cur_bs,
                            det_conf_thres=cfg["det_conf_thres"],
                            id_strip=cfg["site_root"],
                            show_paths=True,
                            path_log_every=cfg["path_log_every"],
                            path_log_mode=cfg["path_log_mode"],
                        ), cur_bs
                    return detection_model_local.batch_image_detection(
                        folder_path,
                        batch_size=cur_bs,
                        det_conf_thres=cfg["det_conf_thres"],
                        id_strip=cfg["site_root"],
                    ), cur_bs
                except Exception as exc:
                    if not (cfg["auto_batch_shrink"] and _is_oom(exc) and cur_bs > cfg["oom_retry_min_batch"]):
                        raise
                    new_bs = max(cfg["oom_retry_min_batch"], cur_bs // 2)
                    cur_bs = new_bs
                    torch.cuda.empty_cache()

        t0 = time.time()
        results, used_bs = _run_batch_detection_local(cfg["batch_size"])
        pw_utils.save_detection_json(
            results,
            json_file,
            categories=detection_model_local.CLASS_NAMES,
            exclude_category_ids=[],
            exclude_file_path=cfg["site_root"],
        )
        sep_msg = pw_utils.detection_folder_separation(
            json_file,
            cfg["site_root"],
            cfg["output_path"],
            float(cfg["threshold"]),
            output_subdir=cfg["site_name"],
            copy_mode=cfg["copy_mode"],
            preserve_relative_paths=True,
        )
        dt = time.time() - t0
        queue.put({
            "status": "ok",
            "folder": folder_path,
            "gpu_id": gpu_id,
            "images": img_count,
            "json_file": json_file,
            "batch_size": used_bs,
            "seconds": f"{dt:.2f}",
            "message": sep_msg,
        })

        del results
        torch.cuda.empty_cache()
        del detection_model_local
    except Exception as exc:
        queue.put({"status": "error", "folder": folder_path, "gpu_id": gpu_id, "error": str(exc)})


def worker_loop(gpu_id: int, task_queue, result_queue, cfg: dict[str, Any]) -> None:
    """Worker loop that pulls folders from a queue and processes them on one GPU."""
    prefix = f"cuda:{gpu_id}"
    log_handle = None
    tracker = _PathTracker()
    stop_event = threading.Event()
    heartbeat_thread = None
    heartbeat_seconds = int(cfg.get("heartbeat_seconds", 0) or 0)
    if cfg.get("capture_worker_stdout", False):
        log_dir = cfg.get("worker_log_dir") or os.path.join(cfg["output_root"], "logs")
        os.makedirs(log_dir, exist_ok=True)
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"gpu{gpu_id}_pid{os.getpid()}_{start_ts}.log")
        log_handle = open(log_path, "a", buffering=1, encoding="utf-8")
        sys.stdout = _TrackingWriter(log_handle, tracker)
        sys.stderr = _TrackingWriter(log_handle, tracker)
        result_queue.put({"status": "worker_log", "gpu_id": gpu_id, "log_path": log_path})
        _log("Worker started", prefix=prefix)
        _log(f"Config: model_family={cfg.get('model_family')} model_version={cfg.get('model_version')} ", prefix=prefix)
        _log(
            "Config: batch_size={bs} det_conf_thres={conf} threshold={thr} copy_mode={cm} resume_skip={rs}"
            .format(
                bs=cfg.get("batch_size"),
                conf=cfg.get("det_conf_thres"),
                thr=cfg.get("threshold"),
                cm=cfg.get("copy_mode"),
                rs=cfg.get("resume_skip"),
            ),
            prefix=prefix,
        )
        if heartbeat_seconds > 0:
            heartbeat_thread = threading.Thread(
                target=_heartbeat_loop,
                args=(stop_event, tracker, heartbeat_seconds, prefix),
                daemon=True,
            )
            heartbeat_thread.start()
    else:
        _log("Worker started", prefix=prefix)

    try:
        while True:
            folder_path = task_queue.get()
            if folder_path is None:
                break
            process_folder_on_gpu(folder_path, gpu_id, cfg, result_queue)
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()
