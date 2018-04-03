# Reference: https://svds.com/jupyter-notebook-best-practices-for-data-science/
import os
import subprocess as sp
from executor import execute
#from subprocess import check_call

def post_save(model, os_path, contents_manager):
    """post-save hook for converting notebooks to .py scripts"""
    if model['type'] != 'notebook':
        return # only do this for notebooks
    d, fname = os.path.split(os_path)

    # fname now has no extension
    fname = fname.replace('.ipynb', '')

    # create subdirectory to store new files if needed
    subdir = d + '/' + fname
    if not os.path.isdir(subdir):
        execute('mkdir ' + subdir.replace(' ', '\ '))
        #sp.check_call(['mkdir', subdir])
    execute('jupyter nbconvert --to script ' + fname + '.ipynb', directory=d)
    execute('jupyter nbconvert --to html ' + fname + '.ipynb', directory=d)
    execute('for f in '+fname+'.*; do mv "$f" "'+subdir+'/$f"; done;', directory=d)
    execute('cp '+subdir.replace(' ', '\ ')+'/'+fname+'.ipynb '+fname+'.ipynb', directory=d)
    execute('rm -f '+subdir.replace(' ', '\ ')+'/'+fname+'.ipynb', directory=d)
    execute('rm -rf Untitled*/', directory=d)

c.FileContentsManager.post_save_hook = post_save
