# Plant Disease Model Training Guide

This guide will help you train your own plant disease detection model using the PlantVillage dataset.

## Prerequisites

Make sure you have all required dependencies installed:

```bash
pip install tensorflow numpy pillow matplotlib opencv-python
```

## Step 1: Download the Dataset

You have several options to get the PlantVillage dataset:

### Option A: Kaggle (Recommended)
1. Go to: https://www.kaggle.com/datasets/emmarex/plantdisease
2. Click "Download" (you'll need a Kaggle account)
3. Extract the downloaded zip file

### Option B: Alternative Source
- GitHub: https://github.com/spMohanty/PlantVillage-Dataset
- Download and extract the dataset

## Step 2: Organize the Dataset

After downloading, organize your data directory as follows:

```
smart farming/
├── data/
│   └── plantvillage/
│       ├── Apple___Apple_scab/
│       │   ├── image1.jpg
│       │   ├── image2.jpg
│       │   └── ...
│       ├── Apple___Black_rot/
│       ├── Apple___Cedar_apple_rust/
│       ├── Apple___healthy/
│       ├── Blueberry___healthy/
│       ├── Cherry___Powdery_mildew/
│       ├── Cherry___healthy/
│       └── ... (other disease classes)
├── models/
├── utils/
└── train_model.py
```

**Important:** 
- Each disease class should be in its own folder
- Folder names should match the class names (e.g., `Apple___Apple_scab`)
- Images should be in JPG, JPEG, or PNG format

## Step 3: Configure Training Parameters

Open `train_model.py` and adjust these parameters if needed:

```python
IMG_SIZE = 224          # Image size (224x224 pixels)
BATCH_SIZE = 32         # Number of images per batch
EPOCHS = 50             # Number of training epochs
LEARNING_RATE = 0.001   # Learning rate for optimizer
```

You can also choose between two model architectures:

1. **Transfer Learning (Recommended)**: Uses pre-trained MobileNetV2
   - Faster training
   - Better accuracy with less data
   - Set `use_transfer_learning=True`

2. **Custom CNN**: Trains from scratch
   - More customizable
   - Requires more data and time
   - Set `use_transfer_learning=False`

## Step 4: Start Training

Run the training script:

```bash
python train_model.py
```

### What Happens During Training:

1. **Data Loading**: Loads and preprocesses images from the dataset
2. **Data Augmentation**: Applies random transformations (rotation, flip, zoom) to increase dataset variety
3. **Model Building**: Creates the neural network architecture
4. **Training**: Trains the model for specified epochs
5. **Validation**: Evaluates model performance on validation set
6. **Checkpointing**: Saves the best model based on validation accuracy
7. **Visualization**: Generates training history plots

### Training Output:

The script will create:
- `models/plant_disease_model.h5` - Final trained model
- `models/checkpoints/model_checkpoint.h5` - Best model checkpoint
- `training_history.png` - Accuracy and loss plots

## Step 5: Monitor Training

During training, you'll see:
- Current epoch and batch progress
- Training accuracy and loss
- Validation accuracy and loss
- Learning rate adjustments
- Best model checkpoints

Example output:
```
Epoch 1/50
100/100 [==============================] - 45s 450ms/step
loss: 1.2345 - accuracy: 0.6789 - val_loss: 0.9876 - val_accuracy: 0.7234
```

### Callbacks:

1. **ModelCheckpoint**: Saves the best model based on validation accuracy
2. **EarlyStopping**: Stops training if validation loss doesn't improve for 10 epochs
3. **ReduceLROnPlateau**: Reduces learning rate when validation loss plateaus

## Step 6: Evaluate Results

After training completes, check:

1. **Final Metrics**: Displayed in the console
   - Training accuracy
   - Validation accuracy
   - Best epoch

2. **Training History Plot**: `training_history.png`
   - Shows accuracy and loss curves
   - Helps identify overfitting or underfitting

### Good Training Signs:
- ✅ Validation accuracy > 85%
- ✅ Training and validation curves are close
- ✅ No large gap between training and validation accuracy

### Issues to Watch For:
- ⚠️ **Overfitting**: Training accuracy >> Validation accuracy
  - Solution: Add more dropout, reduce model complexity
- ⚠️ **Underfitting**: Both accuracies are low
  - Solution: Increase model complexity, train longer
- ⚠️ **Unstable Training**: Accuracy fluctuates wildly
  - Solution: Reduce learning rate, increase batch size

## Step 7: Use Your Trained Model

Once training is complete, your model is automatically saved and ready to use!

The Streamlit app (`app.py`) will automatically load your trained model from `models/plant_disease_model.h5`.

Just run:
```bash
streamlit run app.py
```

## Training Tips

### For Better Accuracy:
1. **Use more data**: The more images, the better
2. **Balance classes**: Ensure each disease class has similar number of images
3. **Use transfer learning**: Pre-trained models often work better
4. **Train longer**: Increase epochs if model is still improving
5. **Fine-tune**: After initial training, unfreeze some base layers and train with lower learning rate

### For Faster Training:
1. **Reduce image size**: Use 128x128 instead of 224x224
2. **Increase batch size**: If you have enough GPU memory
3. **Use fewer epochs**: Start with 20-30 epochs for testing
4. **Use transfer learning**: Much faster than training from scratch

### GPU Acceleration:
If you have an NVIDIA GPU with CUDA:
```bash
pip install tensorflow-gpu
```
This can speed up training by 10-50x!

## Troubleshooting

### "Data directory not found"
- Make sure the dataset is in `data/plantvillage/`
- Check folder structure matches the guide

### "Out of memory" error
- Reduce `BATCH_SIZE` (try 16 or 8)
- Reduce `IMG_SIZE` (try 128)
- Close other applications

### Low accuracy
- Train for more epochs
- Use transfer learning
- Check if data is properly organized
- Ensure images are clear and properly labeled

### Training is too slow
- Use GPU if available
- Reduce image size
- Increase batch size
- Use transfer learning

## Advanced: Training on More Classes

To train on the full PlantVillage dataset (38+ classes):

1. Download the complete dataset
2. Update `PLANT_CLASSES` in `utils/model_loader.py`
3. Add disease information to `PLANT_DISEASES_INFO`
4. Run training script - it will automatically detect all classes

## Need Help?

If you encounter issues:
1. Check the error message carefully
2. Verify dataset organization
3. Try reducing batch size or image size
4. Check TensorFlow/GPU installation

Happy Training! 🌱🚀
