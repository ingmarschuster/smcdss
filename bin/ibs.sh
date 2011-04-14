#!/bin/sh
PROJECT_FOLDER=Documents/Python/smcdss/src
export PYTHONPATH=$HOME/$PROJECT_FOLDER
SAVE_PWD=$PWD
cd $HOME/$PROJECT_FOLDER
python $HOME/$PROJECT_FOLDER/ibs/exec.py $*
cd $SAVE_PWD
