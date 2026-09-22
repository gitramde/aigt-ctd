"""Copy frozen training algorithms; only import bindings and seed literals change."""
import ast,json,hashlib
from . import ROOT
SOURCES={
 'vendor_supervised':('src/phase4/train.py',{'from .data import':'from src.phase4.data import','from .metrics import':'from src.phase4.metrics import','seed=42,':'seed=cfg[\'seed\'],','seed=cfg[\'seed\'],\n                   selection_partition':'seed=self.cfg[\'seed\'],\n                   selection_partition','_seed42_selection.json':'_selection.json'}),
 'vendor_temporal':('src/temporal/train.py',{'from .data import *':'from src.temporal.data import *'}),
 'vendor_graph':('src/graph/train.py',{'from .common import *':'from src.graph.common import *','from .data import GraphData':'from src.graph.data import GraphData','from .model import EdgeModel,tensors':'from src.graph.model import EdgeModel,tensors'}),
 'vendor_anomaly_data':('src/anomaly/data.py',{'from . import ROOT':'from src.final_runs import ROOT'}),
 'vendor_anomaly':('src/anomaly/train.py',{'from .data import *':'from src.final_runs.vendor_anomaly_data import *'}),
 'vendor_graph_temporal':('src/graph_temporal/train.py',{'from .prepare import *':'from src.graph_temporal.prepare import *','from .model import TemporalHead,batch':'from src.graph_temporal.model import TemporalHead,batch'}),
}
class Seed(ast.NodeTransformer):
    def visit_Constant(self,node):
        return ast.copy_location(ast.Name(id='SEED',ctx=ast.Load()),node) if type(node.value)is int and node.value==42 else node
def transformed(source,replacements):
    for old,new in replacements.items():source=source.replace(old,new)
    tree=Seed().visit(ast.parse(source))
    tree.body=[node for node in tree.body if not isinstance(node,ast.If) and not (isinstance(node,ast.FunctionDef) and node.name in ('run','prepare'))]
    ast.fix_missing_locations(tree)
    return 'SEED = 42\n'+ast.unparse(tree)+'\n'
def main():
    for name,(source,replacements) in SOURCES.items():
        (ROOT/f'src/final_runs/{name}.py').write_text(transformed((ROOT/source).read_text(encoding='utf-8-sig'),replacements),encoding='utf-8')
    print('Separate seed-parameterized training copies built; no fitting')
if __name__=='__main__':main()
