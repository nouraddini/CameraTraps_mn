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


def _warm_cache(folder_path: str, bytes_to_read: int = 1024 * 1024) -> None:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".gif", ".webp"}
    for root, _, files in os.walk(folder_path):
        for f in files:
            if os.path.splitext(f)[1].lower() in exts:
                try:
                    with open(os.path.join(root, f), "rb") as fh:
                        fh.read(bytes_to_read)
                except Exception:
                    continue


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

    def reset(self) -> None:
        with self._lock:
            self._last_path = None


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

        if cfg.get("warm_cache", False):
            _log("Warming OS cache for folder", prefix=f"cuda:{gpu_id}")
            _warm_cache(folder_path, int(cfg.get("warm_cache_bytes", 1048576)))

        folder_leaf = os.path.basename(os.path.normpath(folder_path))
        json_file = os.path.join(cfg["output_root"], f"detection_results__{folder_leaf}.json")

        if cfg["resume_skip"] and os.path.isfile(json_file):
            queue.put({
                "status": "skipped_existing_json",
                "folder": folder_path,
                "json_file": json_file,
                "gpu_id": gpu_id,
            })
            return

        img_count = _count_images(folder_path)
        if cfg["skip_empty"] and img_count == 0:
            queue.put({
                "status": "skipped_empty",
                "folder": folder_path,
                "images": 0,
                "gpu_id": gpu_id,
            })
            return

        if cfg["dry_run"]:
            queue.put({
                "status": "dry_run",
                "folder": folder_path,
                "images": img_count,
                "gpu_id": gpu_id,
            })
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
                            num_workers=cfg.get("num_workers", 0),
                            prefetch_factor=cfg.get("prefetch_factor"),
                            persistent_workers=cfg.get("persistent_workers", False),
                            pin_memory=cfg.get("pin_memory", True),
                            decoder=cfg.get("decoder_backend", "pil"),
                            corrupt_log_path=cfg.get("corrupt_log_path"),
                        ), cur_bs
                    return detection_model_local.batch_image_detection(
                        folder_path,
                        batch_size=cur_bs,
                        det_conf_thres=cfg["det_conf_thres"],
                        id_strip=cfg["site_root"],
                        num_workers=cfg.get("num_workers", 0),
                        prefetch_factor=cfg.get("prefetch_factor"),
                        persistent_workers=cfg.get("persistent_workers", False),
                        pin_memory=cfg.get("pin_memory", True),
                        decoder=cfg.get("decoder_backend", "pil"),
                        corrupt_log_path=cfg.get("corrupt_log_path"),
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
        if cfg.get("defer_copy", False):
            sep_msg = "Deferred copy: JSON saved; no file copying performed."
        else:
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
    current_log_path = None
    tracker = _PathTracker()
    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    stop_event = threading.Event()
    heartbeat_thread = None
    heartbeat_seconds = int(cfg.get("heartbeat_seconds", 0) or 0)
    capture_stdout = cfg.get("capture_worker_stdout", False)
    log_dir = None
    if capture_stdout:
        log_dir = cfg.get("worker_log_dir") or os.path.join(cfg["output_root"], "logs")
        os.makedirs(log_dir, exist_ok=True)

    def _sanitize_folder_leaf(name: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")
        return safe or "folder"

    def _open_folder_log(folder_path: str) -> None:
        nonlocal log_handle, current_log_path
        if not capture_stdout or log_dir is None:
            return
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()
        folder_leaf = os.path.basename(os.path.normpath(folder_path))
        safe_leaf = _sanitize_folder_leaf(folder_leaf)
        start_ts = time.strftime("%Y%m%d_%H%M%S")
        current_log_path = os.path.join(
            log_dir,
            f"gpu{gpu_id}_{safe_leaf}_pid{os.getpid()}_{start_ts}.log",
        )
        log_handle = open(current_log_path, "a", buffering=1, encoding="utf-8")
        tracker.reset()
        sys.stdout = _TrackingWriter(log_handle, tracker)
        sys.stderr = _TrackingWriter(log_handle, tracker)
        result_queue.put({
            "status": "worker_log",
            "gpu_id": gpu_id,
            "log_path": current_log_path,
            "folder": folder_path,
        })
        _log("Worker started", prefix=prefix)
        _log(
            f"Config: model_family={cfg.get('model_family')} model_version={cfg.get('model_version')} ",
            prefix=prefix,
        )
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
        _log(f"Processing folder: {folder_path}", prefix=prefix)

    if heartbeat_seconds > 0:
        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            args=(stop_event, tracker, heartbeat_seconds, prefix),
            daemon=True,
        )
        heartbeat_thread.start()
    if not capture_stdout:
        _log("Worker started", prefix=prefix)

    try:
        while True:
            folder_path = task_queue.get()
            if folder_path is None:
                break
            if capture_stdout:
                _open_folder_log(folder_path)
            process_folder_on_gpu(folder_path, gpu_id, cfg, result_queue)
    finally:
        stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=1)
        if log_handle is not None:
            log_handle.flush()
            log_handle.close()
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr
