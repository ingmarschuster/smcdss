#!/bin/sh
oldPWD=$PWD
cd $HOME/Documents/Python/smcdss/src
python exec.py $*
cd $oldPWD
