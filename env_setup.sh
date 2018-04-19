export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
source setup/virtualenvwrapper.sh
mkvirtualenv pblt
pip3 install numpy
pip3 install Cython
pip3 install scipy
pip3 install conda
conda config --add channels http://conda.anaconda.org/psi4
python -m conda install psi4
pip3 install -r setup/requirements.txt
ipython kernel install --user --name=pblt
lssitepackages
