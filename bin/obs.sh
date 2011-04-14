#!/bin/sh
PROJECT=Documents/Python/smcdss/src
export PYTHONPATH=$HOME/$PROJECT
SAVE_PWD=$PWD
cd $HOME/$PROJECT
python $HOME/$PROJECT/obs/exec.py $*
cd $SAVE_PWD

