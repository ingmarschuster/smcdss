:: starts the python exec file
@echo off
SET ROOTDIR=W:\Documents\Software
:: SET ROOTDIR=D:\Dropbox\Python

SET PYTHONPATH=%ROOTDIR%\smcdss\src
PATH=%ROOTDIR%\portable;%ROOTDIR%\portable\python;%ROOTDIR%\portable\mingw\bin;%PATH%
SET CURRENTDIR=%CD%
cd %PYTHONPATH%
%~dp$PATH:1%ROOTDIR%\portable\python\python.exe obs\exec.py %*
cd %CURRENTDIR%