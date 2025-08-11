from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import base64
import json
import pickle
import os
from PIL import Image
import io
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import List, Dict, Optional
import uvicorn

app = FastAPI(title="AI Sketch Guesser API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class DrawingData(BaseModel):
    image_data: str  # base64 encoded image
    
class TeachingData(BaseModel):
    image_data: str
    label: str
    
class StatsResponse(BaseModel):
    total_drawings: int
    unique_classes: int
    model_status: str
    class_counts: Dict[str, int]
    classes: List[str]

class SketchAI:
    def __init__(self):
        self.model = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=42)
        self.scaler = StandardScaler()
        self.classes = ['circle', 'square', 'triangle', 'star', 'heart', 'house', 'car', 'tree']
        self.training_data = []
        self.training_labels = []
        self.model_trained = False
        self.data_file = 'web_sketch_data.pkl'
        
        self.load_training_data()
        self.train_initial_model()
        
    def base64_to_image_data(self, base64_str: str) -> np.ndarray:
        """Convert base64 image to 28x28 normalized array"""
        try:
            # Remove data URL prefix if present
            if base64_str.startswith('data:image'):
                base64_str = base64_str.split(',')[1]
                
            # Decode base64
            image_bytes = base64.b64decode(base64_str)
            
            # Open with PIL
            img = Image.open(io.BytesIO(image_bytes))
            
            # Convert to grayscale and resize to 28x28
            img = img.convert('L')
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            
            # Convert to numpy array and normalize
            img_array = np.array(img)
            img_array = 255 - img_array  # Invert (black on white to white on black)
            img_array = img_array / 255.0  # Normalize to 0-1
            
            return img_array.flatten()
            
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")
    
    def predict(self, image_data: str) -> dict:
        if not self.model_trained:
            return {
                "predictions": [],
                "success": False,
                "message": "Model not trained yet. Please teach me some drawings!"
            }
        
        try:
            img_data = self.base64_to_image_data(image_data)
            
            # Check if image is empty
            if np.sum(img_data) < 0.01:
                return {
                    "predictions": [],
                    "success": False,
                    "message": "Draw something first!"
                }
            
            # Make prediction
            img_data_scaled = self.scaler.transform([img_data])
            probabilities = self.model.predict_proba(img_data_scaled)[0]
            
            # Get top 3 predictions
            top_indices = np.argsort(probabilities)[::-1][:3]
            
            predictions = []
            for idx in top_indices:
                confidence = float(probabilities[idx] * 100)
                class_name = str(self.model.classes_[idx])
                predictions.append({
                    'class': class_name,
                    'confidence': confidence,
                    'emoji': self.get_emoji(class_name)
                })
            
            return {
                "predictions": predictions,
                "success": True,
                "message": "Prediction successful!"
            }
            
        except Exception as e:
            return {
                "predictions": [],
                "success": False,
                "message": f"Prediction error: {str(e)}"
            }
    
    def teach(self, image_data: str, label: str) -> dict:
        try:
            img_data = self.base64_to_image_data(image_data)
            
            if np.sum(img_data) < 0.01:
                return {"success": False, "message": "Please draw something first!"}
            
            # Add to training data
            self.training_data.append(img_data.tolist())
            self.training_labels.append(label.lower())
            
            # Retrain model
            self.retrain_model()
            
            # Save data
            self.save_training_data()
            
            return {"success": True, "message": f"Thanks for teaching me that this is a {label}!"}
            
        except Exception as e:
            return {"success": False, "message": f"Teaching error: {str(e)}"}
    
    def add_class(self, new_class: str) -> dict:
        new_class = new_class.strip().lower()
        if new_class not in self.classes:
            self.classes.append(new_class)
            self.save_training_data()
            return {"success": True, "message": f"Added '{new_class}' to the list of objects!"}
        else:
            return {"success": False, "message": f"'{new_class}' already exists!"}
    
    def get_stats(self) -> StatsResponse:
        total_drawings = len(self.training_data)
        unique_classes = len(set(self.training_labels)) if self.training_labels else 0
        model_status = "Trained" if self.model_trained else "Not Trained"
        
        class_counts = {}
        for label in self.training_labels:
            class_counts[label] = class_counts.get(label, 0) + 1
            
        return StatsResponse(
            total_drawings=total_drawings,
            unique_classes=unique_classes,
            model_status=model_status,
            class_counts=class_counts,
            classes=self.classes
        )
    
    def get_emoji(self, class_name: str) -> str:
        emoji_map = {
            'circle': '⭕', 'square': '⬜', 'triangle': '🔺', 'star': '⭐',
            'heart': '❤️', 'house': '🏠', 'car': '🚗', 'tree': '🌳',
            'cat': '🐱', 'dog': '🐶', 'flower': '🌸', 'sun': '☀️'
        }
        return emoji_map.get(class_name.lower(), '❓')
    
    def retrain_model(self):
        if len(self.training_data) < 2:
            return
            
        try:
            X = np.array(self.training_data)
            y = np.array(self.training_labels)
            
            # Scale the data
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            
            # Train model
            self.model.fit(X_scaled, y)
            self.model_trained = True
            
        except Exception as e:
            print(f"Training error: {e}")
    
    def train_initial_model(self):
        if len(self.training_data) == 0:
            self.create_synthetic_data()
            
        if len(self.training_data) > 0:
            self.retrain_model()
    
    def create_synthetic_data(self):
        np.random.seed(42)
        
        for class_name in self.classes[:4]:
            for _ in range(5):
                if class_name == 'circle':
                    data = self.create_circle_pattern()
                elif class_name == 'square':
                    data = self.create_square_pattern()
                elif class_name == 'triangle':
                    data = self.create_triangle_pattern()
                else:
                    data = np.random.random(784) * 0.3
                    
                self.training_data.append(data.tolist())
                self.training_labels.append(class_name)
    
    def create_circle_pattern(self):
        img = np.zeros((28, 28))
        center = 14
        radius = 8
        for i in range(28):
            for j in range(28):
                dist = np.sqrt((i - center)**2 + (j - center)**2)
                if abs(dist - radius) < 2:
                    img[i, j] = 1.0
        return img.flatten()
        
    def create_square_pattern(self):
        img = np.zeros((28, 28))
        img[6:22, 6:8] = 1.0
        img[6:22, 20:22] = 1.0
        img[6:8, 6:22] = 1.0
        img[20:22, 6:22] = 1.0
        return img.flatten()
        
    def create_triangle_pattern(self):
        img = np.zeros((28, 28))
        for i in range(14):
            start = 14 - i // 2
            end = 14 + i // 2
            img[20 - i, start:end] = 1.0
        return img.flatten()
    
    def save_training_data(self):
        try:
            data = {
                'training_data': self.training_data,
                'training_labels': self.training_labels,
                'classes': self.classes
            }
            with open(self.data_file, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def load_training_data(self):
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.training_data = data.get('training_data', [])
                    self.training_labels = data.get('training_labels', [])
                    self.classes = data.get('classes', self.classes)
        except Exception as e:
            print(f"Error loading data: {e}")

# Initialize AI instance
sketch_ai = SketchAI()

# API Routes
@app.get("/")
async def read_root():
    return {"message": "AI Sketch Guesser API", "version": "1.0.0"}

@app.post("/predict")
async def predict_drawing(drawing: DrawingData):
    """Predict what the drawing represents"""
    result = sketch_ai.predict(drawing.image_data)
    return result.dict() if hasattr(result, 'dict') else result

@app.post("/teach")
async def teach_drawing(teaching: TeachingData):
    """Teach the AI a new drawing"""
    return sketch_ai.teach(teaching.image_data, teaching.label)

@app.post("/add-class")
async def add_new_class(class_data: dict):
    """Add a new object class"""
    new_class = class_data.get('class', '')
    return sketch_ai.add_class(new_class)

@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Get training statistics"""
    return sketch_ai.get_stats()

@app.get("/classes")
async def get_classes():
    """Get list of available classes"""
    return {"classes": sketch_ai.classes}

# Serve static files (for the frontend)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve the main HTML page
@app.get("/app", response_class=HTMLResponse)
async def get_app():
    """Serve the main application page"""
    try:
        with open("static/index.html", "r") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(
            content="<h1>Frontend not found</h1><p>Please ensure the frontend files are in the 'static' directory.</p>",
            status_code=404
        )

if __name__ == "__main__":
    # Create static directory if it doesn't exist
    os.makedirs("static", exist_ok=True)
    
    print("🎨 AI Sketch Guesser API starting...")
    print("📱 Web app will be available at: http://localhost:8000/app")
    print("📖 API docs available at: http://localhost:8000/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
