export WORKON_HOME=~/Envs
mkdir -p $WORKON_HOME
source setup/virtualenvwrapper.sh
mkvirtualenv pblt
pip3 install numpy
pip3 install -r setup/requirements.txt
lssitepackages
