"""Выполнить сохранённый notebook 02, записывая outputs после каждой ячейки."""
from pathlib import Path
import argparse
import asyncio
import sys
import nbformat
from nbclient import NotebookClient

sys.stdout.reconfigure(encoding='utf-8')
if sys.platform=='win32': asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reports-only',action='store_true',
    help='Чистое ядро: конфигурация, настройки MLP и отчёты с диска; без какого-либо fit.')
args=parser.parse_args()
path=root/'notebooks/02_model_training.ipynb'
notebook=nbformat.read(path,as_version=4)
def save(cell,cell_index,**kwargs):
    nbformat.write(notebook,path)
    print('Completed cell',cell_index,flush=True)
    for output in cell.get('outputs',[]):
        if output.output_type=='stream': print(output.text,flush=True)
client=NotebookClient(notebook,timeout=None,kernel_name='python3',
    resources={'metadata':{'path':str(root)}},on_cell_executed=save)
try:
    if args.reports_only:
        # Не исполняем ни одну ячейку fit, независимо от значений флагов в конфигурации.
        first_code=next(i for i,c in enumerate(notebook.cells) if c.cell_type=='code')
        selected=[i for i,c in enumerate(notebook.cells) if c.cell_type=='code' and
            (i==first_code or c.source.startswith('mlp_numeric,mlp_categorical=') or
             'disk_reports_independent_of_technical' in c.metadata.get('tags',[]))]
        with client.setup_kernel():
            for index in selected: client.execute_cell(notebook.cells[index],index)
        notebook.metadata['reporting_validation']={
            'clean_kernel':True,'executed_cell_indices':selected,
            'technical_cells_executed':False,'any_fit_executed':False}
    else:
        client.execute()
finally: nbformat.write(notebook,path)
print('Training notebook completed.',flush=True)
