"""Добавить и выполнить только новые ячейки; старую подготовку не перезапускать."""
from pathlib import Path
import asyncio
import sys
import nbformat
from nbclient import NotebookClient
from quality_notebook_cells import quality_cells

sys.stdout.reconfigure(encoding='utf-8')
if sys.platform=='win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
root=Path(__file__).resolve().parents[1]
path=root/'notebooks/01_data_preparation.ipynb'
notebook=nbformat.read(path,as_version=4)
original=[cell for cell in notebook.cells if 'data_quality_v1' not in cell.metadata.get('tags',[])]
notebook.cells=original+quality_cells()
nbformat.write(notebook,path)
start=len(original)

def save(cell,cell_index,**kwargs):
    nbformat.write(notebook,path)
    print('Completed quality cell',cell_index,flush=True)
    for out in cell.get('outputs',[]):
        if out.output_type=='stream':
            print(out.text,flush=True)

client=NotebookClient(notebook,timeout=1800,kernel_name='python3',
    resources={'metadata':{'path':str(root)}},on_cell_executed=save)
count=max(cell.execution_count or 0 for cell in original if cell.cell_type=='code')
try:
    with client.setup_kernel():
        for i in range(start,len(notebook.cells)):
            if notebook.cells[i].cell_type=='code':
                count+=1
                client.execute_cell(notebook.cells[i],i,execution_count=count)
finally:
    nbformat.write(notebook,path)
print('Quality supplement completed; original cells preserved.',flush=True)
