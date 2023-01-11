@echo off

rem Creating a Python virtual environment
python -m venv venv

call .\venv\Scripts\activate

pip install -r requirements.txt

python main.py -s dataset

pause
