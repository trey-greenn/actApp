from fastapi import FastAPI, File, UploadFile, HTTPException, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Optional
import uuid

from app.models.keras_model import train_model, preprocess_data
from app.utils.file_utils import save_upload_file_temp, remove_file
from app.utils.data_validation import validate_csv_data

app = FastAPI(title="AML Actimize Tuning API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development - restrict this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for MVP
uploads_folder = "uploads"
results_storage = {}
training_status = {}

# Create uploads folder if it doesn't exist
os.makedirs(uploads_folder, exist_ok=True)

@app.get("/")
def read_root():
    return {"message": "AML Actimize Tuning API"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # Validate file extension
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are supported")
    
    try:
        # Generate a unique ID for this upload
        file_id = str(uuid.uuid4())
        
        # Save the file temporarily
        file_path = save_upload_file_temp(file, uploads_folder, file_id)
        
        # Read the CSV file to validate its content
        try:
            df = pd.read_csv(file_path)
            validation_result = validate_csv_data(df)
            
            if not validation_result["is_valid"]:
                # Remove the invalid file
                remove_file(file_path)
                raise HTTPException(status_code=400, detail=validation_result["error"])
                
        except Exception as e:
            # Remove the file if reading fails
            remove_file(file_path)
            raise HTTPException(status_code=400, detail=f"Error reading CSV file: {str(e)}")
        
        # Store the file path for later retrieval
        results_storage[file_id] = {
            "file_path": file_path,
            "filename": file.filename,
            "status": "uploaded"
        }
        
        return {"fileId": file_id, "filename": file.filename}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

def run_training(file_id: str, parameters: Dict):
    try:
        # Update status to training
        training_status[file_id] = {"status": "training", "progress": 0}
        
        # Get the file path
        file_path = results_storage[file_id]["file_path"]
        
        # Read the data
        df = pd.read_csv(file_path)
        
        # Preprocess the data
        X_train, X_test, y_train, y_test, feature_names = preprocess_data(
            df, 
            train_split=parameters.get("trainSplit", 0.8)
        )
        
        # Update progress
        training_status[file_id] = {"status": "training", "progress": 30}
        
        # Train the model
        metrics, confusion_matrix, flagged_indices = train_model(
            X_train, X_test, y_train, y_test,
            learning_rate=parameters.get("learningRate", 0.001),
            epochs=parameters.get("epochs", 10),
            batch_size=parameters.get("batchSize", 32),
            hidden_layers=parameters.get("hiddenLayers", [64, 32])
        )
        
        # Update progress
        training_status[file_id] = {"status": "training", "progress": 90}
        
        # Get flagged transactions
        test_df = df.iloc[X_test.index]
        flagged_transactions = test_df.iloc[flagged_indices].to_dict(orient="records")
        
        # Store the results
        results = {
            "metrics": metrics,
            "confusionMatrix": confusion_matrix,
            "flaggedTransactions": flagged_transactions
        }
        
        # Save the results
        results_storage[file_id]["results"] = results
        results_storage[file_id]["status"] = "completed"
        
        # Update status to completed
        training_status[file_id] = {"status": "completed", "progress": 100}
        
    except Exception as e:
        # Update status to failed
        training_status[file_id] = {"status": "failed", "progress": 0, "error": str(e)}
        raise

@app.post("/train")
async def train(background_tasks: BackgroundTasks, fileId: str = Form(...), parameters: Optional[str] = Form("{}")):
    if fileId not in results_storage:
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        parameters_dict = json.loads(parameters)
        
        # Start training in the background
        background_tasks.add_task(run_training, fileId, parameters_dict)
        
        # Return a training ID (using the same fileId for simplicity in MVP)
        return {"trainingId": fileId}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Training failed to start: {str(e)}")

@app.get("/training-status/{training_id}")
async def get_training_status(training_id: str):
    if training_id not in training_status:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    return training_status[training_id]

@app.get("/training-results/{training_id}")
async def get_training_results(training_id: str):
    if training_id not in results_storage:
        raise HTTPException(status_code=404, detail="Training results not found")
    
    if "results" not in results_storage[training_id]:
        raise HTTPException(status_code=400, detail="Training has not completed yet")
    
    return results_storage[training_id]["results"]

@app.get("/download-results/{training_id}")
async def download_results(training_id: str):
    if training_id not in results_storage:
        raise HTTPException(status_code=404, detail="Training results not found")
    
    if "results" not in results_storage[training_id]:
        raise HTTPException(status_code=400, detail="Training has not completed yet")
    
    # Get the flagged transactions
    flagged_transactions = results_storage[training_id]["results"]["flaggedTransactions"]
    
    # Convert to DataFrame and save to CSV
    flagged_df = pd.DataFrame(flagged_transactions)
    result_file_path = os.path.join(uploads_folder, f"{training_id}_flagged.csv")
    flagged_df.to_csv(result_file_path, index=False)
    
    return FileResponse(
        path=result_file_path,
        filename="flagged_transactions.csv",
        media_type="text/csv"
    )

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}