"""Sequential seed-42 development runs, validation freeze, then held-out evaluation."""
import subprocess
import sys
from .data import MODEL_NAMES,build_training_cache

def run():
    build_training_cache()
    for name in MODEL_NAMES:
        subprocess.run([sys.executable,'-m','src.phase4.train','--model',name],check=True)
    subprocess.run([sys.executable,'-m','src.phase4.evaluate'],check=True)
    for name in MODEL_NAMES:
        subprocess.run([sys.executable,'-m','src.phase4.evaluate','--model',name],check=True)
    subprocess.run([sys.executable,'-m','src.phase4.report'],check=True)

if __name__=='__main__': run()
