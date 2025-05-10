@echo off
echo Setting up Python environment...
python -m venv venv
call venv\Scripts\activate
echo Installing requirements...
pip install -r requirements.txt
echo Setup complete.
pause
