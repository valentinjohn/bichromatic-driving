@echo off

echo Creating virtual environment...
python -m venv venv

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Installing dependencies...
py -m pip install --upgrade pip
py -m pip install -r requirements.txt

echo Download data...
py download_data.py

echo Running script for Figure 1...
py figure1.py
echo Figure 1 plotted and saved.

echo Running script for Figure 2...
py figure2.py
echo Figure 2 plotted and saved.

echo Running script for Figure 3...
py figure3.py
echo Figure 3 plotted and saved.

echo Running script for Figure 4...
py figure4.py
echo Figure 4 plotted and saved.

echo Running script for Figure S1...
py figureS1.py
echo Figure S1 plotted and saved.

echo Running script for Figure S2...
py figureS2.py
echo Figure S2 plotted and saved.

echo Running script for Figure S3...
py figureS3.py
echo Figure S3 plotted and saved.

echo Running script for Figure S4...
py figureS4.py
echo Figure S4 plotted and saved.

echo Running script for Figure S5...
py figureS5.py
echo Figure S5 plotted and saved.

echo All done!
pause
