# Add this at the top of the file, after the existing imports
import tensorflow as tf
import os

# First define the plant classes (all 38 classes from training)
PLANT_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)',
    'Peach___Bacterial_spot',
    'Peach___healthy',
    'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Raspberry___healthy',
    'Soybean___healthy',
    'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch',
    'Strawberry___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# Add the disease information dictionary
PLANT_DISEASES_INFO = {
    'Apple___Apple_scab': {
        'name': 'Apple Scab',
        'symptoms': 'Dark brown or olive green spots on leaves and fruits',
        'solutions': [
            'Remove and destroy infected leaves',
            'Apply fungicides in early spring',
            'Maintain good air circulation by proper pruning',
            'Plant disease-resistant apple varieties'
        ]
    },
    'Apple___Black_rot': {
        'name': 'Black Rot',
        'symptoms': 'Purple spots on leaves, rotting fruit with dark rings',
        'solutions': [
            'Remove infected fruit and cankers',
            'Prune out dead or diseased branches',
            'Apply fungicides during growing season',
            'Maintain proper tree spacing'
        ]
    },
    'Apple___Cedar_apple_rust': {
        'name': 'Cedar Apple Rust',
        'symptoms': 'Bright orange-yellow spots on leaves and fruits',
        'solutions': [
            'Remove nearby cedar trees (alternate host)',
            'Apply preventive fungicides',
            'Plant resistant varieties',
            'Maintain good air circulation'
        ]
    },
    'Apple___healthy': {
        'name': 'Healthy Apple Plant',
        'solutions': ['Continue regular maintenance', 'Monitor for early signs of disease']
    },
    'Blueberry___healthy': {
        'name': 'Healthy Blueberry Plant',
        'solutions': ['Maintain soil pH between 4.5-5.5', 'Regular watering and mulching']
    },
    'Cherry___Powdery_mildew': {
        'name': 'Cherry Powdery Mildew',
        'symptoms': 'White powdery coating on leaves and stems',
        'solutions': [
            'Improve air circulation',
            'Apply fungicides at first sign of disease',
            'Remove infected leaves and branches',
            'Avoid overhead watering'
        ]
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'name': 'Cherry Powdery Mildew',
        'symptoms': 'White powdery coating on leaves and stems',
        'solutions': [
            'Improve air circulation',
            'Apply fungicides at first sign of disease',
            'Remove infected leaves and branches',
            'Avoid overhead watering'
        ]
    },
    'Cherry_(including_sour)___healthy': {
        'name': 'Healthy Cherry Plant',
        'solutions': ['Regular pruning and maintenance', 'Monitor for disease symptoms']
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'name': 'Corn Gray Leaf Spot',
        'symptoms': 'Rectangular gray to tan lesions on leaves',
        'solutions': [
            'Plant resistant varieties',
            'Crop rotation',
            'Apply fungicides if needed',
            'Remove crop debris'
        ]
    },
    'Corn_(maize)___Common_rust_': {
        'name': 'Corn Common Rust',
        'symptoms': 'Circular to elongate reddish-brown pustules on leaves',
        'solutions': [
            'Plant resistant hybrids',
            'Apply fungicides during early infection',
            'Monitor fields regularly'
        ]
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'name': 'Corn Northern Leaf Blight',
        'symptoms': 'Long grayish-green or tan lesions on leaves',
        'solutions': [
            'Use resistant hybrids',
            'Crop rotation with non-host crops',
            'Tillage to bury crop debris',
            'Apply fungicides if severe'
        ]
    },
    'Corn_(maize)___healthy': {
        'name': 'Healthy Corn Plant',
        'solutions': ['Maintain proper spacing', 'Regular monitoring', 'Adequate fertilization']
    },
    'Grape___Black_rot': {
        'name': 'Grape Black Rot',
        'symptoms': 'Circular tan spots on leaves, black mummified berries',
        'solutions': [
            'Remove mummified berries',
            'Prune for air circulation',
            'Apply fungicides preventively',
            'Sanitation is critical'
        ]
    },
    'Grape___Esca_(Black_Measles)': {
        'name': 'Grape Esca (Black Measles)',
        'symptoms': 'Tiger-stripe pattern on leaves, berry spotting',
        'solutions': [
            'Prune out dead wood',
            'Avoid large pruning wounds',
            'No effective chemical control',
            'Remove severely infected vines'
        ]
    },
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': {
        'name': 'Grape Leaf Blight',
        'symptoms': 'Brown spots with dark margins on leaves',
        'solutions': [
            'Improve air circulation',
            'Remove infected leaves',
            'Apply copper-based fungicides',
            'Avoid overhead irrigation'
        ]
    },
    'Grape___healthy': {
        'name': 'Healthy Grape Plant',
        'solutions': ['Proper pruning', 'Good air circulation', 'Regular monitoring']
    },
    'Orange___Haunglongbing_(Citrus_greening)': {
        'name': 'Citrus Greening (HLB)',
        'symptoms': 'Yellow shoots, mottled leaves, lopsided bitter fruit',
        'solutions': [
            'Remove infected trees immediately',
            'Control psyllid vectors',
            'Plant disease-free nursery stock',
            'No cure available - prevention is key'
        ]
    },
    'Peach___Bacterial_spot': {
        'name': 'Peach Bacterial Spot',
        'symptoms': 'Small dark spots on leaves and fruit',
        'solutions': [
            'Plant resistant varieties',
            'Apply copper sprays',
            'Prune for air circulation',
            'Avoid overhead irrigation'
        ]
    },
    'Peach___healthy': {
        'name': 'Healthy Peach Plant',
        'solutions': ['Regular pruning', 'Proper fertilization', 'Monitor for pests']
    },
    'Pepper,_bell___Bacterial_spot': {
        'name': 'Pepper Bacterial Spot',
        'symptoms': 'Small dark spots on leaves and fruit',
        'solutions': [
            'Use disease-free seeds',
            'Copper-based bactericides',
            'Crop rotation',
            'Remove infected plants'
        ]
    },
    'Pepper,_bell___healthy': {
        'name': 'Healthy Pepper Plant',
        'solutions': ['Adequate spacing', 'Proper watering', 'Regular fertilization']
    },
    'Potato___Early_blight': {
        'name': 'Potato Early Blight',
        'symptoms': 'Dark brown spots with concentric rings on leaves',
        'solutions': [
            'Apply fungicides preventively',
            'Crop rotation',
            'Remove infected plant debris',
            'Maintain plant vigor'
        ]
    },
    'Potato___Late_blight': {
        'name': 'Potato Late Blight',
        'symptoms': 'Water-soaked spots on leaves, white mold on undersides',
        'solutions': [
            'Apply fungicides immediately',
            'Plant certified seed potatoes',
            'Destroy infected plants',
            'Avoid overhead irrigation'
        ]
    },
    'Potato___healthy': {
        'name': 'Healthy Potato Plant',
        'solutions': ['Hill soil around plants', 'Regular watering', 'Monitor for diseases']
    },
    'Raspberry___healthy': {
        'name': 'Healthy Raspberry Plant',
        'solutions': ['Prune old canes', 'Mulch well', 'Adequate water']
    },
    'Soybean___healthy': {
        'name': 'Healthy Soybean Plant',
        'solutions': ['Crop rotation', 'Proper spacing', 'Monitor for pests']
    },
    'Squash___Powdery_mildew': {
        'name': 'Squash Powdery Mildew',
        'symptoms': 'White powdery coating on leaves',
        'solutions': [
            'Plant resistant varieties',
            'Apply fungicides',
            'Improve air circulation',
            'Remove infected leaves'
        ]
    },
    'Strawberry___Leaf_scorch': {
        'name': 'Strawberry Leaf Scorch',
        'symptoms': 'Purple spots on leaves that turn brown',
        'solutions': [
            'Remove infected leaves',
            'Apply fungicides',
            'Plant resistant varieties',
            'Improve air circulation'
        ]
    },
    'Strawberry___healthy': {
        'name': 'Healthy Strawberry Plant',
        'solutions': ['Mulch around plants', 'Regular watering', 'Remove old leaves']
    },
    'Tomato___Bacterial_spot': {
        'name': 'Tomato Bacterial Spot',
        'symptoms': 'Small dark spots on leaves and fruit',
        'solutions': [
            'Use disease-free transplants',
            'Copper-based sprays',
            'Crop rotation',
            'Avoid overhead watering'
        ]
    },
    'Tomato___Early_blight': {
        'name': 'Tomato Early Blight',
        'symptoms': 'Dark concentric rings on lower leaves',
        'solutions': [
            'Apply fungicides',
            'Mulch to prevent soil splash',
            'Remove infected leaves',
            'Stake plants for air circulation'
        ]
    },
    'Tomato___Late_blight': {
        'name': 'Tomato Late Blight',
        'symptoms': 'Large brown blotches on leaves and stems',
        'solutions': [
            'Apply fungicides immediately',
            'Remove infected plants',
            'Avoid overhead watering',
            'Plant resistant varieties'
        ]
    },
    'Tomato___Leaf_Mold': {
        'name': 'Tomato Leaf Mold',
        'symptoms': 'Yellow spots on upper leaf surface, fuzzy mold below',
        'solutions': [
            'Improve air circulation',
            'Reduce humidity',
            'Apply fungicides',
            'Remove infected leaves'
        ]
    },
    'Tomato___Septoria_leaf_spot': {
        'name': 'Tomato Septoria Leaf Spot',
        'symptoms': 'Small circular spots with dark borders on leaves',
        'solutions': [
            'Remove infected leaves',
            'Apply fungicides',
            'Mulch to prevent splash',
            'Crop rotation'
        ]
    },
    'Tomato___Spider_mites Two-spotted_spider_mite': {
        'name': 'Tomato Spider Mites',
        'symptoms': 'Stippled leaves, fine webbing, yellowing',
        'solutions': [
            'Spray with water to dislodge mites',
            'Apply insecticidal soap',
            'Use miticides if severe',
            'Maintain plant health'
        ]
    },
    'Tomato___Target_Spot': {
        'name': 'Tomato Target Spot',
        'symptoms': 'Brown spots with concentric rings on leaves',
        'solutions': [
            'Apply fungicides',
            'Remove infected leaves',
            'Improve air circulation',
            'Avoid overhead watering'
        ]
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus',
        'symptoms': 'Upward curling and yellowing of leaves',
        'solutions': [
            'Control whitefly vectors',
            'Remove infected plants',
            'Use reflective mulches',
            'Plant resistant varieties'
        ]
    },
    'Tomato___Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus',
        'symptoms': 'Mottled light and dark green on leaves',
        'solutions': [
            'Remove infected plants',
            'Disinfect tools',
            'Control aphid vectors',
            'Use virus-free seeds'
        ]
    },
    'Tomato___healthy': {
        'name': 'Healthy Tomato Plant',
        'solutions': ['Stake or cage plants', 'Consistent watering', 'Regular fertilization']
    }
}

# Keep the existing load_model function
def load_model(model_path='models/plant_disease_model.h5'):
    """
    Load the pre-trained plant disease classification model
    """
    try:
        # In a real application, you would load your trained model here
        # For demo purposes, we'll create a dummy model
        if not os.path.exists(os.path.dirname(model_path)):
            os.makedirs(os.path.dirname(model_path))
            
        if not os.path.exists(model_path):
            # Create a simple model for demo purposes
            # In a real application, you would load your trained model
            model = tf.keras.Sequential([
                tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                tf.keras.layers.MaxPooling2D(2, 2),
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(128, activation='relu'),
                tf.keras.layers.Dense(len(PLANT_CLASSES), activation='softmax')
            ])
            model.compile(optimizer='adam',
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy'])
            model.save(model_path)
        else:
            model = tf.keras.models.load_model(model_path)
            
        return model, PLANT_CLASSES
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None
