"""Resume prepared Phase9 experiment; fitting is conditional on documented coverage."""
import subprocess,sys
from .prepare import OUT,read

def main():
    if not (OUT/'metrics/coverage_gate.json').exists():
        subprocess.run([sys.executable,'-B','-m','src.graph_temporal.prepare'],check=True)
    if not read(OUT/'metrics/coverage_gate.json')['passed']:
        subprocess.run([sys.executable,'-B','-m','src.graph_temporal.report'],check=True)
        return
    subprocess.run([sys.executable,'-B','-m','src.graph_temporal.train','--embed'],check=True)
    for length in (1,4,8):
        subprocess.run([sys.executable,'-B','-m','src.graph_temporal.train','--length',str(length)],check=True)
    subprocess.run([sys.executable,'-B','-m','src.graph_temporal.evaluate'],check=True)
    subprocess.run([sys.executable,'-B','-m','src.graph_temporal.figures'],check=True)
    subprocess.run([sys.executable,'-B','-m','src.graph_temporal.verify'],check=True)
    subprocess.run([sys.executable,'-B','-m','src.graph_temporal.finalize'],check=True)

if __name__=='__main__':main()
