import streamlit as st
from PIL import Image
import google.generativeai as genai
import io

# --- Configuration ---
# Your API key has been added directly to the code as requested.
GEMINI_API_KEY = "AIzaSyDLRM2X8kQRDrm4jkurtx7CJQyKPrq5IlY"
if not GEMINI_API_KEY:
    st.error("Gemini API key is not set. Please add it to the code or use Streamlit secrets.")
    st.stop()

try:
    genai.configure(api_key=GEMINI_API_KEY)
except Exception as e:
    st.error(f"Failed to configure Gemini API: {e}")
    st.stop()


# --- Helper Function ---
def get_gemini_vision_info(image_data, language):
    """
    Analyzes an image using the Gemini Vision model and returns a descriptive solution
    in the specified language.
    """
    try:
        # Using the latest Gemini 1.5 Flash model for best performance.
        # Note: This requires an up-to-date 'google-generativeai' library.
        vision_model = genai.GenerativeModel('gemini-2.5-flash')
        image = Image.open(io.BytesIO(image_data))
        
        # Base prompt in English for the AI's instructions
        prompt = (
            "You are a world-class plant pathologist and agricultural expert. Analyze the provided image of a plant.\n\n"
            "Your task is to:\n"
            "1.  **Identify the Plant:** First, identify the type of plant (e.g., Tomato, Rose, Apple).\n"
            "2.  **Assess Plant Health:** Determine if the plant is healthy or showing signs of disease or pest infestation.\n"
            "3.  **Provide a Detailed Report:** Based on your assessment, provide a comprehensive guide formatted in Markdown.\n\n"
            "   - If the plant is **diseased or has pests**:\n"
            "       - **Diagnosis:** Clearly state the name of the disease or pest.\n"
            "       - **Disease/Pest Overview:** A brief, clear description of the issue.\n"
            "       - **Common Symptoms:** A bulleted list of the key symptoms.\n"
            "       - **Treatment Methods:** Provide both organic and chemical treatment options.\n"
            "       - **Prevention Strategies:** Actionable steps to prevent the problem from recurring.\n\n"
            "   - If the plant is **healthy**:\n"
            "       - **Confirmation:** State that the plant appears to be healthy.\n"
            "       - **Plant Overview:** A brief description of the plant.\n"
            "       - **Ideal Growing Conditions:** Information on sunlight, soil, and water needs.\n"
            "       - **General Care Tips:** Include fertilization, pruning, and pest control advice.\n\n"
            "Provide a confident and conclusive analysis. Do not be conversational."
        )

        # Add the language translation instruction to the prompt
        language_instruction = f"\n\n**IMPORTANT:** You must generate the entire final report in the {language} language."
        full_prompt = prompt + language_instruction

        response = vision_model.generate_content([full_prompt, image])
        return response.text
    except Exception as e:
        # This error message is crucial. It tells the user how to fix their environment.
        if "404" in str(e) and ("is not found for API version" in str(e) or "is not supported" in str(e)):
            return (
                "**CRITICAL ERROR: Your `google-generativeai` library is OUTDATED.**\n\n"
                "The code is correct, but your computer's environment is not.\n\n"
                "**TO FIX THIS, YOU MUST DO THE FOLLOWING:**\n"
                "1. **Stop this app.**\n"
                "2. **Open your terminal.**\n"
                "3. **Run this command exactly:** `pip install --upgrade google-generativeai`\n"
                "4. **Restart the app.**\n\n"
                f"--- \n*Detailed Error: {e}*"
            )
        return f"An error occurred during image analysis: {e}"

# --- Streamlit App UI ---
st.set_page_config(page_title="Plant Disease Recognition", layout="wide")

# Sidebar for navigation
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "Disease Recognition", "About"])

# --- Page Implementations ---

# Home Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease & Care System! 🌿🔍
    
    Our mission is to help you identify plant diseases and learn how to care for your plants effectively. Upload an image of a plant, and our system will analyze it to detect diseases and provide expert advice in your chosen language.

    ### How It Works
    1. **Navigate to Recognition:** Go to the **Disease Recognition** page from the sidebar.
    2. **Select Language:** Choose your preferred language (English, Hindi, or Gujarati).
    3. **Upload Image:** Upload an image of a plant you are concerned about.
    4. **Get Solution:** Our AI expert (powered by Google Gemini) provides a detailed report in your selected language.

    ### Why Choose Us?
    - **Multilingual Support:** Get expert advice in English, Hindi, and Gujarati.
    - **Vision-Powered:** Get analysis on any plant image without needing pre-trained models.
    - **User-Friendly:** Simple and intuitive interface for a seamless experience.
    
    ---
    Click on the **Disease Recognition** page in the sidebar to get started!
    """)

# About Page
elif app_mode == "About":
    st.header("About The Project")
    st.markdown("""
    #### Technology Stack
    This application is powered entirely by Google's Gemini models.

    - **Backend & UI:** Streamlit
    - **AI Vision Analysis:** The **Gemini 1.5 Flash** model is used for real-time image analysis. It identifies plants, detects diseases, and provides comprehensive reports in multiple languages.

    This project leverages state-of-the-art multimodal AI to provide a flexible and powerful solution for plant disease management and general plant care.
    """)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Plant-Care and Disease Recognition")
    
    st.markdown("Upload an image for a complete AI-powered analysis in your preferred language.")

    # Input: Language Selection
    language = st.selectbox("Select Output Language", ["English", "Hindi", "Gujarati"])

    # Input: Image Uploader
    image_file = st.file_uploader("Upload an image of your plant...", type=["jpg", "jpeg", "png"])
    
    if st.button("Analyze Image"):
        if image_file is not None:
            image_data = image_file.getvalue()
            st.image(image_data, caption="Uploaded Image", use_container_width=True)
            
            with st.spinner(f"Analyzing image and generating report in {language}..."):
                # Pass the selected language to the helper function
                solution = get_gemini_vision_info(image_data, language)
                st.markdown("---")
                st.subheader("💡 AI Vision Analysis Report")
                st.markdown(solution)
        else:
            st.warning("Please upload an image to analyze.")

