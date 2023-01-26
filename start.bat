@echo off

rem Creating a Python virtual environment
rem if it doesn't exist

if exist ".\venv\" (
	goto :run
)

python -m venv venv
call .\venv\Scripts\activate
.\venv\Scripts\python.exe -m pip install -r requirements.txt

:run

call .\venv\Scripts\activate
python main.py -s dataset

pause
