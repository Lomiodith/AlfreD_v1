import glob
import os
import tempfile


def clean_temp_audio():
    temp_dir = tempfile.gettempdir()
    removed_count = 0
    
    for f in glob.glob(os.path.join(temp_dir, "*.wav")):
        try:
            os.remove(f)
            removed_count += 1
        except Exception as e:
            print(f"Warning: Could not remove {f}: {e}")
    
    if removed_count > 0:
        print(f"Cleaned up {removed_count} temporary .wav file(s).")