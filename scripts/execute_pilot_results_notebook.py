"""Исполнить notebook04 в чистом Python namespace и сохранить реальные outputs.

Расчёты bootstrap уже выполнены CLI. Notebook исполняет все свои code cells с
RUN_ANALYSIS=False, читает готовые результаты и не создаёт новых повторов.
--backend jupyter запускает отдельное ядро; backend и факт запуска ядра
явно сохраняются в metadata. Python namespace не требует Jupyter/ZMQ.
"""
from pathlib import Path
import argparse
import json
import os
import sys

import nbformat

from execute_research_pilot_notebook import execute_namespace

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend',choices=['jupyter','python_namespace'],default='python_namespace')
    args=parser.parse_args()
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    os.chdir(ROOT)
    for name in ['TEMP','TMP','IPYTHONDIR','JUPYTER_RUNTIME_DIR']:
        folder=ROOT/'tmp'/'evaluation_runtime'/name.lower()
        folder.mkdir(parents=True,exist_ok=True)
        os.environ[name]=str(folder)
    output=ROOT/'reports/pilot_20k_5config/evaluation'
    summary=json.loads((output/'evaluation_summary.json').read_text(encoding='utf-8'))
    assert summary['completed_bootstrap']==1000 and summary['new_fit_calls']==0
    path=ROOT/'notebooks/04_pilot_results_analysis.ipynb'
    notebook=nbformat.read(path,as_version=4)
    configuration=[cell for cell in notebook.cells if 'evaluation_configuration' in cell.metadata.get('tags',[])]
    assert len(configuration)==1 and 'RUN_ANALYSIS=False' in configuration[0].source
    selected=[i for i,cell in enumerate(notebook.cells) if cell.cell_type=='code']
    execution={'backend':args.backend,'clean_interpreter':True,'clean_kernel':False,
        'status':'running','selected_cells':selected,'executed_cells':[],
        'analysis_computation_source':'completed CLI evaluation; saved1000 bootstrap checkpoints reused as reports',
        'new_fit_calls':0,'new_predict_calls':0,'new_bootstrap_replicates':0,
        'interpreter':sys.executable,'completed_analysis_utc':summary['completed_utc']}
    def save(cell,cell_index,**kwargs):
        if any(item.output_type=='error' for item in cell.outputs):
            execution['failed_cell']=cell_index
        elif cell_index not in execution['executed_cells']:
            execution['executed_cells'].append(cell_index)
        notebook.metadata['evaluation_execution']=execution
        nbformat.write(notebook,path)
        print(f'Saved executed cell {cell_index}',flush=True)
    try:
        if args.backend=='python_namespace':
            execute_namespace(notebook,selected,save)
        else:
            import asyncio
            from nbclient import NotebookClient
            if sys.platform=='win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            client=NotebookClient(notebook,timeout=120,startup_timeout=30,kernel_name='python3',
                resources={'metadata':{'path':str(ROOT)}},on_cell_executed=save)
            client.km=client.create_kernel_manager()
            client.km.kernel_spec.argv[0]=sys.executable
            with client.setup_kernel():
                execution['clean_kernel']=True
                for index in selected:
                    client.execute_cell(notebook.cells[index],index)
        execution['status']='completed'
    except BaseException as exc:
        execution['status']='failed'
        execution['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        notebook.metadata['evaluation_execution']=execution
        nbformat.validate(notebook)
        nbformat.write(notebook,path)
    print('Notebook04 executed; no new bootstrap, fit or predict.',flush=True)


if __name__=='__main__':
    main()
