:: starts the python exec file
@echo off
SET PYTHONDIR=W:\Documents\Python
# SET PYTHONDIR=D:\Dropbox\Python
SET PYTHONPATH=%PYTHONDIR%\smcdss\src
PATH=%PYTHONDIR%\portable;%PYTHONDIR%\portable\App;%PYTHONDIR%\mingw\bin;%PATH%
SET CURRENTDIR=%CD%
cd %PYTHONDIR%\smcdss\src
%~dp$PATH:1%PYTHONDIR%\portable\App\python.exe ibs\exec.py %*
cd %CURRENTDIR%
