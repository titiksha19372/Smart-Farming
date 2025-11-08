import streamlit as st
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from auth import verify_user, register_user

# Import our utilities
from utils.model_loader import load_model, PLANT_DISEASES_INFO
from utils.image_processor import preprocess_image, get_plant_disease_label

# Set page config
st.set_page_config(
    page_title="AgriScan - Plant Disease Detection",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS to remove white container
st.markdown("""
<style>
.css-18e3th9 {
    padding: 0 !important;
    background-color: transparent !important;
}
.css-1d391kg {
    padding: 0 !important;
}
</style>
""", unsafe_allow_html=True)



# Custom CSS for professional styling with animations
def load_css():
    # Add background image directly
    page_bg_img = '''
    <style>
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1500076656116-558758c991c1?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1471&q=80");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-color: rgba(0, 0, 0, 0.5);
        z-index: -1;
    }
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)
    
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: transparent;
        animation: fadeIn 0.5s ease-in;
    }
    
    /* Text color adjustments for better visibility */
    h1, h2, h3, h4, h5, h6, p, li {
        color: white !important;
        text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.8);
    }
    
    /* Special color for main heading */
    h1 {
        color: #4CAF50 !important;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.9);
    }
    
    /* Keyframe animations */
    @keyframes fadeIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInFromLeft {
        from {
            opacity: 0;
            transform: translateX(-30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes slideInFromRight {
        from {
            opacity: 0;
            transform: translateX(30px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    @keyframes scaleIn {
        from {
            opacity: 0;
            transform: scale(0.9);
        }
        to {
            opacity: 1;
            transform: scale(1);
        }
    }
    
    @keyframes pulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
    }
    
    /* Header styling with animation */
    h1 {
        color: #2c3e50;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        font-weight: 700;
        padding-bottom: 10px;
        animation: fadeIn 0.6s ease-out;
    }
    
    h2, h3 {
        color: #34495e;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        animation: fadeIn 0.7s ease-out;
    }
    

    
    /* Card styling with transitions */
    .stAlert {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        animation: scaleIn 0.4s ease-out;
        transition: all 0.3s ease;
    }
    
    .stAlert:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Button styling with enhanced transitions */
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        border: none;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        animation: fadeIn 0.5s ease-out;
        background-color: rgba(39, 174, 96, 0.9);
        color: white;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px) scale(1.02);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
        background-color: rgba(46, 204, 113, 0.9);
    }
    
    .stButton>button:active {
        transform: translateY(-1px) scale(0.98);
        transition: all 0.1s ease;
    }
    
    /* Form styling with transitions */
    .stTextInput>div>div>input {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 10px;
        transition: all 0.3s ease;
        animation: slideInFromLeft 0.5s ease-out;
    }
    
    .stTextInput>div>div>input:focus {
        border-color: #27ae60;
        box-shadow: 0 0 0 3px rgba(39, 174, 96, 0.2);
        transform: scale(1.01);
    }
    
    /* File uploader styling */
    [data-testid="stFileUploader"] {
        animation: scaleIn 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    /* Metric styling with animation */
    [data-testid="stMetricValue"] {
        font-size: 28px;
        font-weight: 700;
        color: #27ae60;
    }
    
    [data-testid="stMetric"] {
        animation: fadeIn 0.6s ease-out;
        transition: all 0.3s ease;
    }
    
    [data-testid="stMetric"]:hover {
        transform: scale(1.05);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2c3e50 0%, #34495e 100%);
        animation: slideInFromLeft 0.4s ease-out;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }
    
    /* Image styling with transition */
    img {
        transition: all 0.3s ease;
        animation: scaleIn 0.6s ease-out;
    }
    
    img:hover {
        transform: scale(1.02);
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.2);
    }
    
    /* Progress bar animation */
    .stProgress > div > div > div {
        transition: all 0.5s ease;
        animation: slideInFromLeft 0.8s ease-out;
    }
    
    /* Info box styling */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
        animation: fadeIn 0.5s ease-out;
        transition: all 0.3s ease;
    }
    
    .info-box:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    
    /* Feature card styling with enhanced animations */
    .feature-card {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        height: 100%;
        animation: scaleIn 0.6s ease-out;
    }
    
    .feature-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
    }
    
    /* Column animations */
    [data-testid="column"] {
        animation: fadeIn 0.6s ease-out;
    }
    
    /* Form container animation */
    [data-testid="stForm"] {
        animation: scaleIn 0.5s ease-out;
    }
    
    /* Spinner animation enhancement */
    .stSpinner > div {
        animation: pulse 1s ease-in-out infinite;
    }
    
    /* Success/Error/Warning messages with slide animation */
    .element-container:has(.stSuccess),
    .element-container:has(.stError),
    .element-container:has(.stWarning),
    .element-container:has(.stInfo) {
        animation: slideInFromRight 0.5s ease-out;
    }
    
    /* Smooth scroll behavior */
    html {
        scroll-behavior: smooth;
    }
    
    /* Container transitions */
    .stContainer {
        transition: all 0.3s ease;
    }
    
    /* Markdown content animation */
    .stMarkdown {
        animation: fadeIn 0.5s ease-out;
    }
    </style>
    """, unsafe_allow_html=True)

def show_home():
    load_css()
    
    # Hero Section
    st.markdown("""
    <div style='text-align: center; padding: 40px 0;'>
        <h1 style='font-size: 3.5em; margin-bottom: 10px; color: #27ae60;'>🌱 AgriScan</h1>
        <h2 style='font-size: 1.8em; color: #34495e; font-weight: 400;'>AI-Powered Plant Disease Detection</h2>
        <p style='font-size: 1.2em; color: #7f8c8d; margin-top: 20px;'>Protect your crops with cutting-edge artificial intelligence technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Call to Action Buttons
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        subcol1, subcol2 = st.columns(2)
        with subcol1:
            if st.button("🔑 Sign In", use_container_width=True, type="primary"):
                st.session_state.page = "Login"
                st.rerun()
        with subcol2:
            if st.button("📝 Create Account", use_container_width=True):
                st.session_state.page = "Register"
                st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Features Section
    st.markdown("""
    <div style='text-align: center; margin: 40px 0 30px 0;'>
        <h2 style='color: #2c3e50; font-size: 2.2em;'>Why Choose AgriScan?</h2>
        <p style='color: #7f8c8d; font-size: 1.1em;'>Advanced technology for modern agriculture</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size: 3em; margin-bottom: 15px;'>🔬</div>
            <h3 style='color: #27ae60; margin-bottom: 10px;'>AI-Powered Analysis</h3>
            <p style='color: #7f8c8d; line-height: 1.6;'>State-of-the-art deep learning models trained on thousands of plant images for accurate disease identification</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size: 3em; margin-bottom: 15px;'>⚡</div>
            <h3 style='color: #27ae60; margin-bottom: 10px;'>Instant Results</h3>
            <p style='color: #7f8c8d; line-height: 1.6;'>Get real-time disease detection results in seconds, enabling quick decision-making for crop management</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <div style='font-size: 3em; margin-bottom: 15px;'>💡</div>
            <h3 style='color: #27ae60; margin-bottom: 10px;'>Expert Recommendations</h3>
            <p style='color: #7f8c8d; line-height: 1.6;'>Receive detailed treatment solutions and preventive measures backed by agricultural research</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # How It Works Section
    st.markdown("""
    <div style='text-align: center; margin: 50px 0 30px 0;'>
        <h2 style='color: #2c3e50; font-size: 2.2em;'>How It Works</h2>
        <p style='color: #7f8c8d; font-size: 1.1em;'>Simple, fast, and effective</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: #27ae60; color: white; width: 60px; height: 60px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; 
                        font-size: 24px; font-weight: bold;'>1</div>
            <h4 style='color: #2c3e50;'>Register</h4>
            <p style='color: #7f8c8d;'>Create your account</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: #27ae60; color: white; width: 60px; height: 60px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; 
                        font-size: 24px; font-weight: bold;'>2</div>
            <h4 style='color: #2c3e50;'>Upload</h4>
            <p style='color: #7f8c8d;'>Add plant leaf image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: #27ae60; color: white; width: 60px; height: 60px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; 
                        font-size: 24px; font-weight: bold;'>3</div>
            <h4 style='color: #2c3e50;'>Analyze</h4>
            <p style='color: #7f8c8d;'>AI processes the image</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style='text-align: center; padding: 20px;'>
            <div style='background: #27ae60; color: white; width: 60px; height: 60px; border-radius: 50%; 
                        display: flex; align-items: center; justify-content: center; margin: 0 auto 15px; 
                        font-size: 24px; font-weight: bold;'>4</div>
            <h4 style='color: #2c3e50;'>Get Results</h4>
            <p style='color: #7f8c8d;'>View diagnosis & solutions</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='text-align: center; padding: 30px; background: white; border-radius: 10px; margin-top: 40px;'>
        <h3 style='color: #27ae60; margin-bottom: 15px;'>Ready to protect your crops?</h3>
        <p style='color: #7f8c8d; font-size: 1.1em;'>Join thousands of farmers using AI-powered disease detection</p>
    </div>
    """, unsafe_allow_html=True)

def detect_disease():
    load_css()
    
    # Sidebar user info
    st.sidebar.markdown("""
    <div style='padding: 20px; background: rgba(255, 255, 255, 0.1); border-radius: 10px; margin-bottom: 20px;'>
        <h3 style='color: white; margin-bottom: 10px;'>👤 User Profile</h3>
        <p style='color: #ecf0f1; font-size: 1.1em;'><strong>{}</strong></p>
    </div>
    """.format(st.session_state.username), unsafe_allow_html=True)
    
    # Sidebar menu
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Menu")
    
    # Navigation options
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()
    
    if st.sidebar.button("🌱 Disease Detection", use_container_width=True, disabled=True):
        pass  # Already on this page
    
    st.sidebar.markdown("---")
    
    # Logout button in sidebar
    if st.sidebar.button("🚪 Logout", use_container_width=True, type="primary"):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.session_state.page = "home"
        # Clear query parameter on logout
        st.query_params.clear()
        st.rerun()
    
    # Additional info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style='padding: 15px; background: rgba(39, 174, 96, 0.2); border-radius: 8px; margin-top: 20px;'>
        <p style='color: #ecf0f1; font-size: 0.9em; margin: 0;'>
            <strong>💡 Tip:</strong> Upload clear, well-lit images for best results
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("""
    <div style='text-align: center; padding: 20px 0;'>
        <h1 style='color: #27ae60; font-size: 2.8em; border: none; border-bottom: none;'>🌱 Disease Detection Dashboard</h1>
        <p style='color: #7f8c8d; font-size: 1.2em;'>Upload a plant leaf image for AI-powered analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Load model
    model, class_names = load_model()
    
    if model is None:
        st.error("Error loading the model. Please check the model files.")
        return
    
    # File uploader with better styling
    with st.container():
        st.markdown("""
        <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1); margin-bottom: 20px;'>
            <h3 style='color: #2c3e50; margin-top: 0;'>📤 Upload Plant Image</h3>
            <p style='color: #7f8c8d; margin-bottom: 15px;'>Select a clear image of the plant leaf for analysis</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # Create two columns for better layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Display the uploaded image in a smaller size
            st.markdown("### 📷 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption='Plant Leaf Image', width=400)
        
        with col2:
            st.markdown("### 🔍 Analysis")
        
        # Make prediction
        with st.spinner('🔄 Analyzing the plant...'):
            try:
                # Preprocess the image
                processed_image = preprocess_image(uploaded_file)
                
                # Make prediction
                prediction = model.predict(processed_image)
                
                # Get the predicted class and confidence
                predicted_class, confidence = get_plant_disease_label(prediction, class_names)
                
                # Get disease information
                disease_info = PLANT_DISEASES_INFO[predicted_class]
                
                # Display results in the second column
                with col2:
                    st.markdown("#### 📊 Results")
                    
                    # Metrics in a compact layout
                    st.metric("Detected Condition", disease_info['name'])
                    st.metric("Confidence Level", f"{confidence*100:.2f}%")
                    
                    # Display a nice progress bar for confidence
                    st.progress(float(confidence))
                
                # Full width section for detailed information
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
                
                # Show disease information
                if "healthy" in predicted_class.lower():
                    st.success("✅ This plant appears to be healthy!")
                    
                    # Healthy plant maintenance tips
                    st.markdown("### 🌿 Maintenance Tips")
                    st.markdown("""
                    <div style='background: #d4edda; padding: 20px; border-radius: 10px; border-left: 5px solid #28a745;'>
                        <h4 style='color: #155724; margin-top: 0;'>Keep Your Plant Healthy</h4>
                        <ul style='color: #155724; line-height: 1.8;'>
                            <li><strong>Regular Watering:</strong> Maintain consistent soil moisture without overwatering</li>
                            <li><strong>Proper Sunlight:</strong> Ensure adequate light exposure based on plant requirements</li>
                            <li><strong>Nutrient Management:</strong> Apply balanced fertilizers at recommended intervals</li>
                            <li><strong>Pest Monitoring:</strong> Regularly inspect for early signs of pests or diseases</li>
                            <li><strong>Pruning:</strong> Remove dead or damaged leaves to promote healthy growth</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error("⚠️ Disease Detected - Immediate attention required")
                    
                    # Disease Information Section
                    st.markdown("### 🦠 Disease Information")
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.markdown("""
                        <div style='background: #fff3cd; padding: 20px; border-radius: 10px; border-left: 5px solid #ffc107;'>
                            <h4 style='color: #856404; margin-top: 0;'>📋 Disease Details</h4>
                            <p style='color: #856404;'><strong>Condition:</strong> {}</p>
                            <p style='color: #856404;'><strong>Severity:</strong> {}</p>
                        </div>
                        """.format(
                            disease_info['name'],
                            "High" if confidence > 0.8 else "Medium" if confidence > 0.6 else "Low"
                        ), unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown("""
                        <div style='background: #f8d7da; padding: 20px; border-radius: 10px; border-left: 5px solid #dc3545;'>
                            <h4 style='color: #721c24; margin-top: 0;'>🔍 Common Symptoms</h4>
                            <p style='color: #721c24; line-height: 1.6;'>{}</p>
                        </div>
                        """.format(disease_info.get('symptoms', 'Visible signs of disease on leaves, stems, or fruits. Discoloration, spots, wilting, or abnormal growth patterns may be present.')), unsafe_allow_html=True)
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # Treatment Solutions Section
                    st.markdown("### 💊 Treatment & Management Solutions")
                    
                    st.markdown("""
                    <div style='background: #d1ecf1; padding: 25px; border-radius: 10px; border-left: 5px solid #17a2b8;'>
                        <h4 style='color: #0c5460; margin-top: 0;'>Recommended Actions</h4>
                    """, unsafe_allow_html=True)
                    
                    # Display solutions in a detailed format
                    for idx, solution in enumerate(disease_info['solutions'], 1):
                        st.markdown(f"""
                        <div style='margin: 15px 0; padding: 15px; background: white; border-radius: 8px;'>
                            <p style='color: #0c5460; margin: 0;'><strong>Step {idx}:</strong> {solution}</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    # Additional recommendations
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.info("""
                    **⚡ Quick Tips:**
                    - Act quickly to prevent disease spread
                    - Isolate affected plants if possible
                    - Remove and dispose of severely infected plant parts
                    - Improve air circulation around plants
                    - Avoid overhead watering to reduce moisture on leaves
                    - Consider consulting with a local agricultural extension service for severe cases
                    """)
                    
            except Exception as e:
                st.error(f"Error during prediction: {str(e)}")

def show_login():
    load_css()
    
    # Center the login form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #27ae60; font-size: 2.5em;'>🔑 Sign In</h1>
            <p style='color: #7f8c8d; font-size: 1.1em;'>Welcome back to AgriScan</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Login form in a styled container
        with st.container():
            st.markdown("""
            <div style='background: white; padding: 40px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);'>
            """, unsafe_allow_html=True)
            
            with st.form("login_form"):
                st.markdown("<h3 style='color: #2c3e50; margin-bottom: 20px;'>Enter your credentials</h3>", unsafe_allow_html=True)
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                
                st.markdown("<br>", unsafe_allow_html=True)
                submit = st.form_submit_button("Sign In", use_container_width=True, type="primary")
                
                if submit:
                    if not username or not password:
                        st.error("⚠️ Please fill in all fields")
                    elif verify_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.username = username
                        st.session_state.page = "detect"
                        # Set query parameter to persist session
                        st.query_params['user'] = username
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password. Please try again.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
        with col_b:
            if st.button("Create Account →", use_container_width=True):
                st.session_state.page = "Register"
                st.rerun()
        
        st.markdown("""
        <div style='text-align: center; margin-top: 30px; color: #7f8c8d;'>
            <p>Don't have an account? Click 'Create Account' to get started</p>
        </div>
        """, unsafe_allow_html=True)

def show_register():
    load_css()
    
    # Center the registration form
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div style='text-align: center; padding: 20px 0;'>
            <h1 style='color: #27ae60; font-size: 2.5em;'>📝 Create Account</h1>
            <p style='color: #7f8c8d; font-size: 1.1em;'>Join AgriScan today</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Registration form in a styled container
        with st.container():
            st.markdown("""
            <div style='background: white; padding: 40px; border-radius: 15px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);'>
            """, unsafe_allow_html=True)
            
            with st.form("register_form"):
                st.markdown("<h3 style='color: #2c3e50; margin-bottom: 20px;'>Fill in your details</h3>", unsafe_allow_html=True)
                
                username = st.text_input("Username", placeholder="Choose a unique username")
                email = st.text_input("Email Address", placeholder="your.email@example.com")
                password = st.text_input("Password", type="password", placeholder="Minimum 6 characters")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Re-enter your password")
                
                st.markdown("<br>", unsafe_allow_html=True)
                
                # Password requirements info
                st.info("ℹ️ Password must be at least 6 characters long")
                
                submit = st.form_submit_button("Create Account", use_container_width=True, type="primary")
                
                if submit:
                    if not username or not email or not password:
                        st.error("⚠️ All fields are required!")
                    elif "@" not in email or "." not in email:
                        st.error("⚠️ Please enter a valid email address")
                    elif len(password) < 6:
                        st.error("⚠️ Password must be at least 6 characters long!")
                    elif password != confirm_password:
                        st.error("⚠️ Passwords do not match!")
                    else:
                        if register_user(username, password, email):
                            st.success("✅ Registration successful! Please sign in.")
                            st.balloons()
                            st.session_state.page = "Login"
                            st.rerun()
                        else:
                            st.error("❌ Username already exists. Please choose another one.")
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Navigation buttons
        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("← Back to Home", use_container_width=True):
                st.session_state.page = "home"
                st.rerun()
        with col_b:
            if st.button("Sign In →", use_container_width=True):
                st.session_state.page = "Login"
                st.rerun()
        
        st.markdown("""
        <div style='text-align: center; margin-top: 30px; color: #7f8c8d;'>
            <p>Already have an account? Click 'Sign In' to access your dashboard</p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Check query parameters for session persistence
    query_params = st.query_params
    
    # Initialize session state
    if 'logged_in' not in st.session_state:
        # Try to restore session from query params
        if 'user' in query_params:
            st.session_state.logged_in = True
            st.session_state.username = query_params['user']
            st.session_state.page = "detect"
        else:
            st.session_state.logged_in = False
    
    if 'page' not in st.session_state:
        st.session_state.page = "home"
    
    if 'username' not in st.session_state:
        st.session_state.username = None
    
    # Navigation
    if st.session_state.page == "home":
        show_home()
    elif st.session_state.page == "Login":
        show_login()
    elif st.session_state.page == "Register":
        show_register()
    elif st.session_state.page == "detect":
        if not st.session_state.logged_in:
            st.warning("Please log in to access the detection tool.")
            st.session_state.page = "Login"
            st.rerun()
        else:
            detect_disease()
    else:
        st.session_state.page = "home"
        st.rerun()

if __name__ == "__main__":
    main()
