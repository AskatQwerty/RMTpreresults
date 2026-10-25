"""Обновить отчётные ячейки notebook 03 без исполнения ячейки обучения.

По умолчанию используется чистый Python-процесс с общим namespace ячеек.
Настоящее ядро Jupyter запускается только с --backend jupyter.
"""
from pathlib import Path
import argparse
import base64
from contextlib import redirect_stderr, redirect_stdout
import io
import os
import sys
import traceback

import nbformat

ROOT=Path(__file__).resolve().parents[1]


def selected_cells(notebook,reports_only=False):
    selected=[]
    for index,cell in enumerate(notebook.cells):
        if cell.cell_type!='code': continue
        tags=cell.metadata.get('tags',[])
        if 'research_pilot_explicit_opt_in' in tags: continue
        if not reports_only or any(tag in tags for tag in
                                   ['pilot_configuration','pilot_disk_reports_independent']):
            selected.append(index)
    return selected


class CellStream(io.TextIOBase):
    """Сохраняет порядок print/stderr/display в outputs текущей ячейки."""
    def __init__(self,cell,name):
        self.cell=cell
        self.name=name

    @property
    def encoding(self):
        return 'utf-8'

    def writable(self):
        return True

    def write(self,text):
        if text:
            outputs=self.cell.outputs
            if outputs and outputs[-1].output_type=='stream' and outputs[-1].name==self.name:
                outputs[-1].text+=text
            else:
                outputs.append(nbformat.v4.new_output('stream',name=self.name,text=text))
        return len(text)

    def flush(self):
        pass


def execute_namespace(notebook,selected,on_cell_executed):
    """Обычный Python exec без Jupyter/ZMQ; отчётные ячейки не имеют magic/await."""
    import IPython.display as ipython_display
    from IPython.core.formatters import DisplayFormatter

    formatter=DisplayFormatter()
    namespace={'__name__':'__main__'}
    original_display=ipython_display.display
    try:
        for count,index in enumerate(selected,1):
            cell=notebook.cells[index]
            cell.outputs=[]
            cell.execution_count=count

            def capture_display(*objects,raw=False,include=None,exclude=None,metadata=None,**kwargs):
                for obj in objects:
                    data,display_metadata=(dict(obj),{}) if raw else formatter.format(
                        obj,include=include,exclude=exclude)
                    if metadata: display_metadata.update(metadata)
                    data={mime:base64.b64encode(value).decode('ascii') if isinstance(value,bytes)
                          else value for mime,value in data.items()}
                    cell.outputs.append(nbformat.v4.new_output('display_data',data=data,
                                                               metadata=display_metadata))

            ipython_display.display=capture_display
            namespace['display']=capture_display
            try:
                with redirect_stdout(CellStream(cell,'stdout')),redirect_stderr(CellStream(cell,'stderr')):
                    exec(compile(cell.source,f'<notebook-cell-{index}>','exec'),namespace)
            except BaseException as exc:
                cell.outputs.append(nbformat.v4.new_output('error',ename=type(exc).__name__,
                    evalue=str(exc),traceback=traceback.format_exception(type(exc),exc,exc.__traceback__)))
                on_cell_executed(cell=cell,cell_index=index)
                raise
            on_cell_executed(cell=cell,cell_index=index)
    finally:
        ipython_display.display=original_display


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports-only',action='store_true')
    parser.add_argument('--backend',choices=['python_namespace','jupyter'],default='python_namespace')
    args=parser.parse_args(argv)
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    for name,folder in [('JUPYTER_RUNTIME_DIR','jupyter_runtime'),('IPYTHONDIR','pilot_ipython'),
                        ('TEMP','pilot_runtime'),('TMP','pilot_runtime')]:
        path=ROOT/'tmp'/folder
        path.mkdir(parents=True,exist_ok=True)
        os.environ[name]=str(path)
    os.chdir(ROOT)
    path=ROOT/'notebooks/03_pilot_20k_5config.ipynb'
    notebook=nbformat.read(path,as_version=4)
    selected=selected_cells(notebook,args.reports_only)
    execution={'backend':args.backend,'status':'running','clean_kernel':False,
               'clean_interpreter':True,'selected_cells':selected,'executed_cells':[],
               'training_cell_executed':False,'interpreter':sys.executable}
    notebook.metadata['report_execution']=execution

    def save(cell,cell_index,**kwargs):
        if any(output.output_type=='error' for output in cell.outputs):
            execution['failed_cell']=cell_index
        elif cell_index not in execution['executed_cells']:
            execution['executed_cells'].append(cell_index)
        notebook.metadata['report_execution']=execution
        nbformat.write(notebook,path)
        print('Saved notebook cell',cell_index,flush=True)

    try:
        if args.backend=='python_namespace':
            execute_namespace(notebook,selected,save)
        else:
            import asyncio
            from nbclient import NotebookClient
            if sys.platform=='win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            client=NotebookClient(notebook,timeout=None,kernel_name='python3',
                resources={'metadata':{'path':str(ROOT)}},on_cell_executed=save)
            client.km=client.create_kernel_manager()
            client.km.kernel_spec.argv[0]=sys.executable
            with client.setup_kernel():
                execution['clean_kernel']=True
                for index in selected: client.execute_cell(notebook.cells[index],index)
        execution['status']='completed'
    except BaseException as exc:
        execution['status']='failed'
        execution['error']=f'{type(exc).__name__}: {exc}'
        raise
    finally:
        notebook.metadata['report_execution']=execution
        nbformat.write(notebook,path)
    print(f'Notebook reports updated with {args.backend}; no fit executed.',flush=True)


if __name__=='__main__':
    main()
