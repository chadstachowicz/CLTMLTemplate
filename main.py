from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from torchvision import datasets,transforms
import torchvision
import json
import os
import io
from PIL import Image
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import TumorClassifier


model = TumorClassifier()
model.load_state_dict(torch.load("model/model-v1"))
model.eval()

transform = transforms.Compose([transforms.Resize(128),
                                 transforms.CenterCrop(128),
                                 transforms.ToTensor()])
classes = ('Glioma', 'Meningioma', 'No Tumor', 'Pituitary Tumor')

tumor_descriptions = [
    {
        "tumor": "Glioma",
        "description": (
            "Gliomas are a type of tumor that starts in the glial cells of the brain or spinal cord. "
            "These tumors can be malignant or benign and are known for their rapid growth and potential to infiltrate nearby brain tissue. "
            "Common symptoms include headaches, seizures, and neurological deficits depending on the tumor's location. It’s essential to consult a medical professional."
        )
    },
    {
        "tumor": "Meningioma",
        "description": (
            "Meningiomas are typically slow-growing tumors that arise from the meninges, the membranes that cover the brain and spinal cord. "
            "Most meningiomas are benign, though some can be atypical or malignant. "
            "Symptoms often develop gradually and may include headaches, vision changes, or seizures depending on the tumor’s size and location. It’s essential to consult a medical professional."
        )
    },
    {
        "tumor": "No Tumor",
        "description": (
            "No tumor indicates that no abnormal growth or mass has been detected in the brain. "
            "This result suggests healthy brain tissue, but it’s essential to consult a medical professional "
            "for further confirmation and analysis."
        )
    },
    {
        "tumor": "Pituitary Tumor",
        "description": (
            "Pituitary tumors develop in the pituitary gland, a small organ at the base of the brain that regulates vital hormones. "
            "These tumors are often benign and categorized as functioning or non-functioning based on their effect on hormone production. "
            "Symptoms may include hormonal imbalances, vision problems, or unexplained fatigue. It’s essential to consult a medical professional."
        )
    }
]

# Define input data schema
class InputData(BaseModel):
    inputs: list

# Initialize FastAPI app
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
# Directory where model JSON files are stored
MODEL_DIRECTORY = "model_info"
@app.get("/")
def read_root():
    return FileResponse("static/index.html")

@app.post("/predict")
async def predict(file: UploadFile = File(...), model_name: str = Query(..., description="The name of the model to use for prediction")):
    model.load_state_dict(torch.load("model/" + model_name))
    model.eval()
    # Read the uploaded image file
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Apply transformations
    transformed_image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        outputs = model(transformed_image)
        _, predicted = torch.max(outputs, 1)
        print('Predicted: ',classes[predicted[0]])
    return {"prediction": classes[predicted[0]], "description": tumor_descriptions[predicted[0]]["description"]}



@app.get("/get-model")
async def get_model(model_name: str):
    """
    Retrieve and parse the JSON file corresponding to the given model name.

    Args:
        model_name (str): The name of the model.

    Returns:
        JSON object: Parsed contents of the model JSON file.
    """
    # Construct the file path based on the model name
    file_path = Path(MODEL_DIRECTORY) / f"{model_name}.json"
    print(file_path)
    # Check if the file exists
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file '{model_name}.json' not found.")

    # Read and parse the JSON file
    try:
        with open(file_path, "r") as file:
            model_data = json.load(file)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Error decoding JSON. Ensure the file is valid JSON.")

    return {"model_data": model_data}

@app.get("/models")
async def list_models():
    models_dir = "./model"  # Replace with the actual path to your models directory
    try:
        # List all files in the directory
        files = os.listdir(models_dir)
        # Filter out directories, if any
        model_files = [file for file in files if os.path.isfile(os.path.join(models_dir, file))]
        return JSONResponse(content=model_files)
    except FileNotFoundError:
        return JSONResponse(content={"error": "Models directory not found."}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)