import sys
import os
import threading
import pystray
from PIL import Image
from AlfreD_v1 import main as alfred_main
from pathlib import Path


def get_base_path():
    """Get base path - works both for dev and PyInstaller bundle."""
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path.cwd()


class AlfredTray:
    def __init__(self):
        self.running = False
        self.alfred_thread = None
        self.icon = None

    def create_icon_image(self):
        return Image.open(get_base_path() / "assets" / "alfred_128.png")

    def start_alfred(self, icon, item):
        if self.running:
            return
        from AlfreD_v1 import stop_event
        stop_event.clear()
        self.running = True
        self.alfred_thread = threading.Thread(target=self._run_alfred, daemon=True)
        self.alfred_thread.start()
        print("Alfred initialized")

    def stop_alfred(self, icon=None, item=None):
        if not self.running:
            return
        from AlfreD_v1 import stop_event
        stop_event.set()
        if self.alfred_thread and self.alfred_thread.is_alive():
            self.alfred_thread.join(timeout=5)
        self.alfred_thread = None
        self.running = False
        print("Alfred stopped")

    def quit_app(self, icon, item):
        self.stop_alfred()
        icon.stop()

    def _run_alfred(self):
        try:
            alfred_main()
        except Exception as e:
            print(f"Encountered an error : {e}")
        finally:
            self.running = False

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
    tray = AlfredTray()
    tray.run()
