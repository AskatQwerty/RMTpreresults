"""Выполнить notebook и сохранять выводы после каждой ячейки, в том числе при ошибке."""
from pathlib import Path
import sys
import asyncio
import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding='utf-8')
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
root = Path(__file__).resolve().parents[1]
path = root/'notebooks/01_data_preparation.ipynb'
nb = nbformat.read(path, as_version=4)

def save_progress(cell, cell_index, **kwargs):
    nbformat.write(nb,path)
    print('Completed cell',cell_index,flush=True)
    for out in cell.get('outputs',[]):
        if out.output_type=='stream':
            print(out.text,flush=True)

client = NotebookClient(nb,timeout=7200,kernel_name='python3',
                        resources={'metadata':{'path':str(root)}},
                        on_cell_executed=save_progress)
try:
    client.execute()
finally:
    nbformat.write(nb,path)
print('Notebook execution complete',flush=True)
