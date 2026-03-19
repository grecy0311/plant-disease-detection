import streamlit as st
import tensorflow as tf
import numpy as np

# ------------------ Model Prediction Function ------------------
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# ------------------ Sidebar ------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# ------------------ Home Page ------------------
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM 🌿")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the **Plant Disease Recognition System**!  
    Upload an image of a plant leaf 🌱, and our AI will identify the disease and suggest treatments.

    ### 🔍 How It Works
    1. Upload a leaf image on the **Disease Recognition** page.
    2. The model analyzes it using a trained CNN model.
    3. Get disease name, description, and recommended treatment.

    ### 💡 Key Features
    - High accuracy with CNN deep learning.
    - Multi-language support for farmers.
    - Real-time image analysis using Gemini API (for future enhancement).

    """)

# ------------------ About Page ------------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
    #### 🌾 Dataset Info
    - Total images: ~87K  
    - Classes: 38 (Healthy + Diseased leaves)  
    - Split: 80% Training, 20% Validation  
    - Source: Augmented PlantVillage dataset  

    #### 🧠 Model Info
    - Model: Convolutional Neural Network (CNN)
    - Accuracy: ~97%  
    - Framework: TensorFlow & Keras  
    """)

# ------------------ Disease Recognition Page ------------------
elif app_mode == "Disease Recognition":
    st.header("🩺 Plant Disease Recognition")
    test_image = st.file_uploader("Upload a Plant Leaf Image:")

    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    # Predict Button
    if st.button("Predict"):
        st.snow()
        st.write("🔍 **Analyzing Image...**")

        result_index = model_prediction(test_image)

        # ------------------ Class Names ------------------
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']

        prediction = class_name[result_index]
        st.success(f"✅ **Model Prediction:** {prediction}")

        # ------------------ Disease Information Dictionary ------------------
        disease_info = {
            'Apple___Apple_scab': {
                'Description': 'A fungal disease causing dark lesions on leaves and fruit.',
                'Treatment': 'Use fungicides like mancozeb or captan; prune infected leaves.'
            },
            'Apple___Black_rot': {
                'Description': 'Fungal infection causing fruit rot and leaf spots.',
                'Treatment': 'Remove mummified fruits and apply copper-based fungicide.'
            },
            'Apple___Cedar_apple_rust': {
                'Description': 'Orange spots on leaves; spreads from nearby cedar trees.',
                'Treatment': 'Remove nearby cedar hosts; apply fungicides containing myclobutanil.'
            },
            'Corn_(maize)___Common_rust_': {
                'Description': 'Reddish-brown pustules appear on leaves due to fungal rust.',
                'Treatment': 'Use resistant hybrids and apply strobilurin fungicides.'
            },
            'Corn_(maize)___Northern_Leaf_Blight': {
                'Description': 'Gray-green lesions on leaves caused by Exserohilum turcicum.',
                'Treatment': 'Rotate crops and use resistant corn varieties.'
            },
            'Potato___Early_blight': {
                'Description': 'Dark spots with concentric rings caused by Alternaria solani.',
                'Treatment': 'Apply fungicides such as chlorothalonil or copper oxychloride.'
            },
            'Potato___Late_blight': {
                'Description': 'Water-soaked spots; rapid leaf death caused by Phytophthora infestans.',
                'Treatment': 'Use fungicides (metalaxyl), destroy infected plants, and avoid overhead irrigation.'
            },
            'Tomato___Early_blight': {
                'Description': 'Circular spots with dark concentric rings on lower leaves.',
                'Treatment': 'Use copper fungicide weekly and rotate crops regularly.'
            },
            'Tomato___Leaf_Mold': {
                'Description': 'Olive-green mold growth under the leaves in humid conditions.',
                'Treatment': 'Improve air circulation and apply chlorothalonil spray.'
            },
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
                'Description': 'Caused by whiteflies; leaves curl upward and yellowing appears.',
                'Treatment': 'Use insect-proof nets and neem oil spray; remove infected plants.'
            },
            'Tomato___healthy': {
                'Description': 'No disease detected — your plant is healthy! 🌿',
                'Treatment': 'Maintain regular watering and sunlight exposure.'
            }
        }

        # ------------------ Display Disease Info ------------------
        if prediction in disease_info:
            st.subheader("🧾 Disease Description")
            st.write(disease_info[prediction]['Description'])
            st.subheader("💊 Recommended Treatment")
            st.write(disease_info[prediction]['Treatment'])
        else:
            st.info("⚠️ Description not available for this disease yet.")

