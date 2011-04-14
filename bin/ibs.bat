:: starts the python exec file
@echo off
set PYTHONPATH=W:\Documents\Python\smcdss\src
PATH=%PATH%;W:\Documents\Python\portable;W:\Documents\Python\portable\App;W:\Documents\Python\mingw\bin
SET CURRENTDIR=%CD%
cd W:\Documents\Python\smcdss\src
%~dp$PATH:1W:\Documents\Python\portable\App\python.exe ibs\exec.py %*
cd %CURRENTDIR%
