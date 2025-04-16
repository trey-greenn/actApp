import os
import shutil
from fastapi import UploadFile

def save_upload_file_temp(upload_file: UploadFile, destination_folder: str, file_id: str) -> str:
    """
    Save the upload file to a temporary destination
    """
    # Create destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)
    
    # Get file extension
    file_extension = os.path.splitext(upload_file.filename)[1]
    
    # Create file path
    file_path = os.path.join(destination_folder, f"{file_id}{file_extension}")
    
    # Write file
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(upload_file.file, buffer)
    
    return file_path

def remove_file(file_path: str) -> None:
    """
    Remove a file if it exists
    """
    if os.path.exists(file_path):
        os.remove(file_path)