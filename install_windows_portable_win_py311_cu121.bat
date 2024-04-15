@echo off

set "requirements_txt=%~dp0\requirements.txt"
set "python_exec=..\..\..\python_embeded\python.exe"

echo Starting to install ComfyUI-InstantMesh...

if exist "%python_exec%" (
    echo Installing with ComfyUI Windows Portable Python Embeded Environment

    "%python_exec%" -s -m pip install -r "%requirements_txt%" 
) else (
    echo ERROR: Cannot find ComfyUI Windows Portable Python Embeded Environment "%python_exec%"
)

echo Install Finished. Press any key to continue...

pause