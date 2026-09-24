import logging
import sys
import threading
from pathlib import Path

import pystray
from PIL import Image

from AlfreD_v2 import main as alfred_main, stop_event
from utils import setup_logging

logger = logging.getLogger(__name__)


def get_base_path():
    """Get base path - works both for dev and PyInstaller bundle."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


class AlfredTray:
    def __init__(self):
        self.alfred_thread = None
        self.icon = None

    def _is_running(self):
        return self.alfred_thread is not None and self.alfred_thread.is_alive()

    def create_icon_image(self):
        return Image.open(get_base_path() / "assets" / "alfred_128.png")

    def start_alfred(self, icon, item):
        if self._is_running():
            return
        stop_event.clear()
        self.alfred_thread = threading.Thread(target=self._run_alfred, daemon=True)
        self.alfred_thread.start()
        logger.info("Alfred initialized")

    def stop_alfred(self, icon=None, item=None):
        if not self._is_running():
            return
        stop_event.set()
        self.alfred_thread.join(timeout=5)
        logger.info(
            "Alfred stopped"
            if not self._is_running()
            else "Alfred is still stopping..."
        )

    def quit_app(self, icon, item):
        self.stop_alfred()
        icon.stop()

    def _run_alfred(self):
        try:
            alfred_main()
        except Exception as e:
            logger.error(f"Encountered an error : {e}")

    def run(self):
        menu = pystray.Menu(
            pystray.MenuItem("Start Alfred", self.start_alfred),
            pystray.MenuItem("Stop Alfred", self.stop_alfred),
            pystray.MenuItem("Quit", self.quit_app),
        )
        self.icon = pystray.Icon(
            "Alfred", self.create_icon_image(), "Alfred - Voice Assistant", menu
        )
        self.icon.run()


if __name__ == "__main__":
    setup_logging()
    tray = AlfredTray()
    tray.run()
