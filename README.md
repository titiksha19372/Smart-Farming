# Plant Disease Detection System

An AI-powered web application that helps identify plant diseases from leaf images using deep learning.

## Features

- Upload plant leaf images for disease detection
- Real-time analysis with confidence scores
- Mobile-responsive web interface
- Sample images for testing
- Detailed results with visual feedback

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. Open your web browser and navigate to `http://localhost:8501`

3. Upload an image of a plant leaf using the file uploader

4. View the analysis results, including the predicted disease and confidence level

## Model    

This application uses a pre-trained deep learning model for plant disease classification. The model has been trained on a dataset of various plant species and their common diseases.

## Dataset

The model was trained on the [PlantVillage Dataset](https://plantvillage.psu.edu/), which contains images of healthy and diseased plant leaves.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Note

This application is for educational and informational purposes only. For actual agricultural decisions, please consult with a professional agronomist or plant pathologist.
