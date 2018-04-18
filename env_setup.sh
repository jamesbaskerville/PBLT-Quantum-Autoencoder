export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
source setup/virtualenvwrapper.sh
mkvirtualenv pblt
pip3 install numpy
pip3 install Cython
pip3 install scipy
pip3 install -r setup/requirements.txt
pip3 install openfermion
pip3 install openfermionpsi4
ipython kernel install --user --name=pblt
lssitepackages
