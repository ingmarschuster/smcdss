#!/bin/sh
PROJECT=Documents/Software/smcdss/src
export PYTHONPATH=$HOME/$PROJECT
SAVE_PWD=$PWD
cd $HOME/$PROJECT
/usr/bin/python2.6 $HOME/$PROJECT/obs/exec.py $*
cd $SAVE_PWD

