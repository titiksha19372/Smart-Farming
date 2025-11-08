"""
Plant Disease Detection Model Training Script
This script trains a CNN model on the PlantVillage dataset for plant disease classification.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 5
LEARNING_RATE = 0.001

# Paths
DATA_DIR = 'data/plantvillage'  # You'll need to download the dataset here
MODEL_SAVE_PATH = 'models/plant_disease_model.h5'
CHECKPOINT_PATH = 'models/checkpoints/model_checkpoint.h5'

# Plant disease classes (from model_loader.py)
PLANT_CLASSES = [
    'Apple___Apple_scab',
    'Apple___Black_rot',
    'Apple___Cedar_apple_rust',
    'Apple___healthy',
    'Blueberry___healthy',
    'Cherry___Powdery_mildew',
    'Cherry___healthy'
]

def create_data_generators(data_dir, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Create training and validation data generators with augmentation
    """
    # Training data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2  # 80% training, 20% validation
    )
    
    # Validation data (only rescaling, no augmentation)
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    # Create training generator
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    # Create validation generator
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=(img_size, img_size),
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

def build_model(num_classes, img_size=IMG_SIZE):
    """
    Build a CNN model for plant disease classification
    Using a custom architecture with transfer learning option
    """
    # Option 1: Custom CNN Architecture
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_size, img_size, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Second Convolutional Block
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Third Convolutional Block
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Fourth Convolutional Block
        layers.Conv2D(256, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Dropout(0.25),
        
        # Flatten and Dense Layers
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def build_transfer_learning_model(num_classes, img_size=IMG_SIZE):
    """
    Build a model using transfer learning with MobileNetV2
    This is often more accurate and trains faster
    """
    # Load pre-trained MobileNetV2 model
    base_model = keras.applications.MobileNetV2(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add custom layers on top
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_training_history(history, save_path='training_history.png'):
    """
    Plot training and validation accuracy/loss
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def train_model(use_transfer_learning=True):
    """
    Main training function
    """
    print("=" * 60)
    print("Plant Disease Detection Model Training")
    print("=" * 60)
    
    # Check if data directory exists
    if not os.path.exists(DATA_DIR):
        print(f"\nERROR: Data directory '{DATA_DIR}' not found!")
        print("\nPlease download the PlantVillage dataset and organize it as follows:")
        print(f"{DATA_DIR}/")
        print("  ├── Apple___Apple_scab/")
        print("  ├── Apple___Black_rot/")
        print("  ├── Apple___Cedar_apple_rust/")
        print("  ├── Apple___healthy/")
        print("  └── ... (other disease classes)")
        print("\nDataset can be downloaded from:")
        print("https://www.kaggle.com/datasets/emmarex/plantdisease")
        return
    
    # Create directories
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    
    # Create data generators
    print("\n[1/5] Loading and preparing data...")
    train_generator, val_generator = create_data_generators(DATA_DIR)
    
    num_classes = len(train_generator.class_indices)
    print(f"Number of classes: {num_classes}")
    print(f"Training samples: {train_generator.samples}")
    print(f"Validation samples: {val_generator.samples}")
    print(f"Class names: {list(train_generator.class_indices.keys())}")
    
    # Build model
    print("\n[2/5] Building model...")
    if use_transfer_learning:
        print("Using Transfer Learning with MobileNetV2")
        model = build_transfer_learning_model(num_classes)
    else:
        print("Using Custom CNN Architecture")
        model = build_model(num_classes)
    
    # Compile model
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("\nModel Summary:")
    model.summary()
    
    # Setup callbacks
    print("\n[3/5] Setting up training callbacks...")
    callbacks = [
        ModelCheckpoint(
            CHECKPOINT_PATH,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # Train model
    print("\n[4/5] Starting training...")
    print(f"Training for {EPOCHS} epochs with batch size {BATCH_SIZE}")
    print("-" * 60)
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model
    print("\n[5/5] Saving model...")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nFinal Training Accuracy: {history.history['accuracy'][-1]:.4f}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.4f}")
    print(f"Final Training Loss: {history.history['loss'][-1]:.4f}")
    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.4f}")
    
    # Find best epoch
    best_epoch = np.argmax(history.history['val_accuracy']) + 1
    best_val_acc = max(history.history['val_accuracy'])
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    return model, history

if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    tf.random.set_seed(42)
    
    # Train the model
    # Set use_transfer_learning=False to use custom CNN instead
    model, history = train_model(use_transfer_learning=True)
