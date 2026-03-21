@echo off
REM Setup and Launch Script for Windows
REM Traffic Prediction + Signal Control + Green Corridor System

echo.
echo ════════════════════════════════════════════════════════════════
echo   Traffic Prediction + Signal Control + Green Corridor System
echo                     Integrated Setup - Windows
echo ════════════════════════════════════════════════════════════════
echo.

REM Step 1: Check Python
echo [1/6] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Please install Python 3.8+
    pause
    exit /b 1
)
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo OK - Python %PYTHON_VERSION% found

REM Step 2: Activate virtual environment
echo.
echo [2/6] Checking virtual environment...
if exist "myenv\" (
    echo OK - Virtual environment found
    call myenv\Scripts\activate.bat
    echo OK - Virtual environment activated
) else (
    echo Creating virtual environment...
    python -m venv myenv
    call myenv\Scripts\activate.bat
    echo OK - Virtual environment created and activated
)

REM Step 3: Install dependencies
echo.
echo [3/6] Installing dependencies...
python -m pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt >nul 2>&1
if errorlevel 1 (
    echo WARNING: Some packages failed to install
) else (
    echo OK - Dependencies installed successfully
)

REM Step 4: Check data files
echo.
echo [4/6] Checking data files...
if not exist "outputs\traffic_predictions_5s.csv" (
    echo WARNING: Traffic predictions not found
    echo   Run: python traffic_prediction_pipeline.py
) else (
    echo OK - Traffic predictions found
)

if not exist "outputs\embeddings.csv" (
    echo WARNING: GNN embeddings not found
    echo   Run: python gnn_embedding_pipeline.py
) else (
    echo OK - GNN embeddings found
)

REM Step 5: Run integration tests
echo.
echo [5/6] Running integration tests...
python test_integration.py
if errorlevel 0 (
    echo OK - Integration tests completed
)

REM Step 6: Launch Streamlit
echo.
echo [6/6] Launching Streamlit dashboard...
echo.
echo OK - Starting application...
echo.
echo Dashboard URL: http://localhost:8501
echo Press Ctrl+C to stop
echo.

streamlit run streamlit_app.py

echo.
echo OK - Application closed
pause
