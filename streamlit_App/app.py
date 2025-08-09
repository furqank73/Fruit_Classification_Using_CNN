import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Fruit & Veg Classifier",
    page_icon="🍎",
    layout="centered",
)

# ---------- ENHANCED CUSTOM CSS ----------
st.markdown("""
<style>
.social-icons {
    text-align: center;
    margin-top: 30px;
}

.social-icons a {
    text-decoration: none;
    color: white;
    font-size: 2rem;
    margin: 0 15px;
    transition: transform 0.3s ease, color 0.3s ease;
}

.social-icons a:hover {
    transform: scale(1.2);
}

.fa-github:hover { color: #333; }
.fa-linkedin:hover { color: #0A66C2; }
.fa-kaggle:hover { color: #20BEFF; }
</style>

<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Hide Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Enhanced background gradient */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        background-size: 400% 400%;
        animation: gradientMove 20s ease infinite;
    }
    
    @keyframes gradientMove {
        0% {background-position: 0% 50%;}
        50% {background-position: 100% 50%;}
        100% {background-position: 0% 50%;}
    }
    
    /* Professional title styling */
    h1 {
        text-align: center;
        font-weight: 800;
        font-size: 3.5rem !important;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9ff 50%, #e8eaff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 8px 32px rgba(0,0,0,0.3);
        margin-bottom: 2rem !important;
        letter-spacing: -0.02em;
    }
    
    /* Enhanced glass card */
    .glass-card {
        background: rgba(255, 255, 255, 0.2);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 24px;
        padding: 2.5rem;
        box-shadow: 
            0 25px 45px -10px rgba(0, 0, 0, 0.2),
            0 10px 20px -5px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .glass-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 
            0 35px 60px -10px rgba(0, 0, 0, 0.25),
            0 20px 40px -5px rgba(0, 0, 0, 0.15);
    }
    
    /* Enhanced prediction text */
    .pred-class {
        font-size: 2.2rem !important;
        font-weight: 700;
        background: linear-gradient(135deg, #4675ab);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-align: center;
        letter-spacing: -0.01em;
    }
    
    .pred-confidence {
        font-size: 1.6rem;
        background: linear-gradient(135deg, #bf5a5a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 600;
        text-align: center;
        letter-spacing: 0.05em;
    }
    
    /* Enhanced upload box */
    .stFileUploader {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        border: 2px dashed rgba(255, 255, 255, 0.4);
        transition: all 0.3s ease;
    }
    
    .stFileUploader:hover {
        border-color: rgba(255, 255, 255, 0.7);
        background: rgba(255, 255, 255, 0.25);
        transform: scale(1.02);
    }
    
    .stFileUploader label {
        font-weight: 600 !important;
        font-size: 1.2rem !important;
        color: #ffffff !important;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Image styling */
    .stImage > div {
        border-radius: 16px;
        overflow: hidden;
        box-shadow: 0 20px 40px rgba(0,0,0,0.2);
    }
    
    /* Column styling */
    .stColumn > div {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin: 0.5rem 0;
    }
    
    /* Mobile responsiveness */
    @media (max-width: 768px) {
        h1 {
            font-size: 2.5rem !important;
        }
        
        .glass-card {
            padding: 1.5rem;
            margin: 0.5rem;
        }
        
        .pred-class {
            font-size: 1.8rem !important;
        }
        
        .pred-confidence {
            font-size: 1.4rem;
        }
    }
    
    /* Success animation */
    @keyframes successPop {
        0% {
            transform: scale(0.8);
            opacity: 0;
        }
        50% {
            transform: scale(1.1);
        }
        100% {
            transform: scale(1);
            opacity: 1;
        }
    }
    
    .glass-card {
        animation: successPop 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('fruit_classification_model.keras')

model = load_model()

# Class names in model order
class_names = [
    'apple', 'banana', 'beetroot', 'bell pepper', 'cabbage', 'capsicum', 'carrot', 'cauliflower',
    'chilli pepper', 'corn', 'cucumber', 'eggplant', 'garlic', 'ginger', 'grapes', 'jalepeno',
    'kiwi', 'lemon', 'lettuce', 'mango', 'onion', 'orange', 'paprika', 'pear', 'peas', 'pineapple',
    'pomegranate', 'potato', 'raddish', 'soy beans', 'spinach', 'sweetcorn', 'sweetpotato',
    'tomato', 'turnip', 'watermelon'
]

# ---------- TITLE ----------
st.title("🍏 Fruit & Vegetable Classifier 🍇")

# ---------- FILE UPLOAD ----------
uploaded_file = st.file_uploader("📸 Upload an image to classify...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    
    # Layout: Two columns (image | prediction)
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(img, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        with st.spinner("🔍 Analyzing your image..."):
            # Preprocess
            img_resized = img.resize((224, 224))
            img_array = np.array(img_resized) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            predictions = model.predict(img_array)
            pred_prob = np.max(predictions)
            pred_class = class_names[np.argmax(predictions)]
        
        # Enhanced glassmorphism card
        st.markdown(f"""
        <div class="glass-card">
            <div class="pred-class">Prediction: {pred_class.title()}</div>
            <div class="pred-confidence">Confidence: {pred_prob*100:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div class="social-icons">
        <a href="https://github.com/furqank73" target="_blank" class="fab fa-github"></a>
        <a href="https://www.linkedin.com/in/furqan-khan-256798268/" target="_blank" class="fab fa-linkedin"></a>
        <a href="https://www.kaggle.com/fkgaming" target="_blank" class="fab fa-kaggle"></a>
        </div>
        """, unsafe_allow_html=True)
