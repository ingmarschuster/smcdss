#!/bin/sh
export PYTHONPATH=$HOME/Documents/Python/smcdss/src
PROJECT=Documents/Python/smcdss/src
SAVE_PWD=$PWD
cd $HOME/$PROJECT
python $HOME/$PROJECT/ibs/exec.py $*
cd $SAVE_PWD
