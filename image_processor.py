import numpy as np
from PIL import Image
import cv2

def preprocess_image(image, target_size=(224, 224)):
    """
    Preprocess the image for model prediction
    """
    if isinstance(image, str):
        # If input is a file path
        img = Image.open(image)
    else:
        # If input is a file upload
        img = Image.open(image)
    
    # Convert to RGB if not already
    img = img.convert('RGB')
    
    # Resize the image
    img = img.resize(target_size)
    
    # Convert to numpy array and normalize
    img_array = np.array(img) / 255.0
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

def get_plant_disease_label(prediction, class_names):
    """
    Convert model prediction to human-readable label
    """
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)
    return class_names[predicted_class], float(confidence)
