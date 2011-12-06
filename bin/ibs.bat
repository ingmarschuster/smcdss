:: starts the python exec file
@echo off
SET PYTHONPATH=%ROOTDIR%\smcdss\src
PATH=%ROOTDIR%\portable;%ROOTDIR%\portable\python;%ROOTDIR%\portable\mingw\bin;%PATH%
SET CURRENTDIR=%CD%
cd %PYTHONPATH%
%~dp$PATH:1%ROOTDIR%\portable\python\python.exe ibs\exec.py %*
cd %CURRENTDIR%