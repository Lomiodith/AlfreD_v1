"""Starts and stops llama.cpp's llama-server for a `llamacpp:` model in LLM_CHAIN."""

import ctypes
import logging
import subprocess
import sys
import time
from ctypes import wintypes
from typing import Optional

import requests

from config import LLAMACPP_ARGS, LLAMACPP_MODEL, LLAMACPP_SERVER, LLAMACPP_URL
from utils import stop_event

logger = logging.getLogger(__name__)

# A first load from a cold disk can take a minute or more.
STARTUP_TIMEOUT = 240

JOB_OBJECT_EXTENDED_LIMIT_INFORMATION = 9
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000


class _BasicLimits(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.c_size_t),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _ExtendedLimits(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _BasicLimits),
        ("IoInfo", ctypes.c_ulonglong * 6),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


def _close_with_this_process(process: subprocess.Popen):
    """Windows: tie `process` to a job that closes when AlfreD's process ends,
    however it ends. A closed console window or a crash skips `stop()`, and the
    orphaned server kept ~8 GB of GPU memory. Returns the job handle, which must
    stay open for as long as the server should live."""
    if sys.platform != "win32":
        return None
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]

    job = kernel32.CreateJobObjectW(None, None)
    limits = _ExtendedLimits()
    limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not (
        job
        and kernel32.SetInformationJobObject(
            job,
            JOB_OBJECT_EXTENDED_LIMIT_INFORMATION,
            ctypes.byref(limits),
            ctypes.sizeof(limits),
        )
        and kernel32.AssignProcessToJobObject(job, int(process._handle))
    ):
        logger.warning(
            f"⚠️ llama-server may outlive AlfreD (job error {ctypes.get_last_error()})"
        )
    return job


def _healthy() -> bool:
    try:
        return requests.get(f"{LLAMACPP_URL}/health", timeout=2).ok
    except requests.RequestException:
        return False


class LlamaCppServer:
    def __init__(self):
        self._process: Optional[subprocess.Popen] = None
        self._job = None

    def ensure_running(self, alias: str):
        """Start llama-server unless one is already answering on LLAMACPP_URL."""
        if _healthy():
            logger.info("🦙 llama-server already running")
            return
        if not (LLAMACPP_SERVER and LLAMACPP_MODEL):
            logger.warning(
                "⚠️ LLAMACPP_SERVER / LLAMACPP_MODEL not set; can't start llama-server"
            )
            return

        host, _, port = LLAMACPP_URL.split("//", 1)[1].partition(":")
        command = [
            LLAMACPP_SERVER,
            "-m",
            LLAMACPP_MODEL,
            "--alias",
            alias,
            "--jinja",
            "--host",
            host,
            "--port",
            port or "8080",
            *LLAMACPP_ARGS,
        ]
        logger.info(f"🦙 Starting llama-server ({alias}); loading can take a minute...")
        self._process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        self._job = _close_with_this_process(self._process)
        deadline = time.time() + STARTUP_TIMEOUT
        while time.time() < deadline and not stop_event.is_set():
            if self._process.poll() is not None:
                logger.error(
                    f"❌ llama-server exited with code {self._process.returncode}"
                )
                self._process = None
                return
            if _healthy():
                logger.info("🦙 llama-server ready")
                return
            time.sleep(1)
        logger.warning("⚠️ llama-server not ready yet; the fallback models will cover")

    def stop(self):
        """Stop the server if AlfreD started it (one started by hand is left alone)."""
        if self._process and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
