# ---------------------------------------------
# run_project.ps1
# Fully setup and run ML pipeline locally
# ---------------------------------------------

# Step 0: Check if venv exists, create if missing
if (-Not (Test-Path ".\venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

# Step 1: Activate virtual environment
Write-Host "Activating virtual environment..."
.\venv\Scripts\Activate.ps1

# Step 2: Upgrade pip
Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

# Step 3: Install dependencies
Write-Host "Installing dependencies..."
pip install -r requirements.txt

# Step 4: Train the model
Write-Host "Training the Logistic Regression model..."
python -m src.train

# Step 5: Run FastAPI backend
Write-Host "Starting FastAPI backend on http://127.0.0.1:8000 ..."
Start-Process "powershell" "uvicorn backend.main:app --reload"

# Step 6: Run Streamlit frontend
Write-Host "Starting Streamlit frontend on http://localhost:8501 ..."
Start-Process "powershell" "streamlit run frontend/app.py"

Write-Host "All processes started. Use browser to access frontend and backend."

__pycache__/
*.pyc
venv/
env/
.ipynb_checkpoints/
data/
*.csv
*.xlsx

*.pkl
logs/

