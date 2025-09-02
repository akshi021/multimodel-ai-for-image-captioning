import streamlit as st
import os
from openai import OpenAI
from PIL import Image
import io
import base64
from pathlib import Path
from dotenv import load_dotenv
# Load environment variables
load_dotenv()
QWEN_API_KEY = os.getenv("QWEN_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client for TTS
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Qwen client
qwen_client = OpenAI(
    api_key=QWEN_API_KEY,
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)

# Function to check Qwen model availability
def check_qwen_model_availability():
    try:
        response = qwen_client.chat.completions.create(
            model="qwen2-vl-7b-instruct",
            messages=[{"role": "user", "content": "Test"}],
            max_tokens=1
        )
        return True, "qwen2-vl-7b-instruct"
    except Exception as e:
        st.warning(f"Qwen Model (qwen2-vl-7b-instruct) not available: {e}")
        try:
            response = qwen_client.chat.completions.create(
                model="qwen-vl-max",
                messages=[{"role": "user", "content": "Test"}],
                max_tokens=1
            )
            return True, "qwen-vl-max"
        except Exception as fallback_e:
            st.error(f"Fallback Qwen Model (qwen-vl-max) not available: {fallback_e}")
            return False, None

# Function to check OpenAI TTS model availability
def check_openai_tts_availability():
    try:
        response = openai_client.audio.speech.create(
            model="tts-1-hd",
            voice="alloy",
            input="Test"
        )
        return True, "tts-1-hd"
    except Exception as e:
        st.error(f"OpenAI TTS model (tts-1-hd) not available: {e}")
        st.warning("TTS functionality is unavailable. The app will display the text description only.")
        return False, None

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function to generate text from image using Qwen
def generate_text_from_image(image, model_name):
    base64_image = encode_image(image)
    try:
        response = qwen_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe the content of the image in detail."
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        }
                    ]
                }
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error generating text from image: {e}")
        return None

# Function to convert text to speech using OpenAI TTS
def text_to_speech(text, voice_type="alloy", tts_model="tts-1-hd"):
    if not tts_model:
        return None
    max_length = 4096
    chunks = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    # Process only the first chunk to avoid combining
    try:
        response = openai_client.audio.speech.create(
            model=tts_model,
            voice=voice_type,
            input=chunks[0]  # Use only the first chunk
        )
        speech_file_path = Path("speech.mp3")
        response.stream_to_file(speech_file_path)
        return speech_file_path
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}")
        return None

# Streamlit app
def main():
    st.set_page_config(page_title="Vision-to-Speech Assistant", page_icon="🤖", layout="centered")
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        :root {
            --primary-color: #00ff9d;
            --background-color: #121212;
            --text-color: #ffffff;
        }
        .stApp {
            background-color: var(--background-color);
            color: var(--text-color);
        }
        .stButton>button {
            background-color: var(--primary-color);
            color: var(--background-color);
        }
        </style>
    """, unsafe_allow_html=True)
    
    st.title("Vision-to-Speech Assistant")
    st.write("Upload an image, and the app will describe it and convert the description to speech (if available).")
    st.info("Note: Only the first 4096 characters of the description will be converted to speech to avoid external dependencies.")
    
    # Check Qwen model availability
    qwen_available, qwen_model = check_qwen_model_availability()
    if not qwen_available:
        st.error("No Qwen vision model available. Please check your API key or available models in Alibaba Cloud Model Studio.")
        return
    
    st.success(f"Using Qwen model: {qwen_model}")
    
    # Check OpenAI TTS model availability
    tts_available, tts_model = check_openai_tts_availability()
    
    # Image upload
    uploaded_image = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])
    
    # Voice selection (only show if TTS is available)
    voice_type = st.selectbox("Choose the voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]) if tts_available else None
    
    if uploaded_image is not None:
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("Generate Description and Speech"):
            with st.spinner("Generating text description..."):
                description = generate_text_from_image(image, qwen_model)
            
            if description:
                st.write("**Generated Description:**")
                st.write(description)
                
                if tts_available:
                    with st.spinner("Converting to speech..."):
                        speech_file_path = text_to_speech(description, voice_type, tts_model)
                    
                    if speech_file_path:
                        audio_file = open(speech_file_path, 'rb')
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                        st.download_button(
                            label="Download Speech",
                            data=audio_bytes,
                            file_name="speech.mp3",
                            mime="audio/mp3"
                        )
                else:
                    st.warning("Text-to-speech is unavailable due to model access restrictions. The description is displayed above.")

if __name__ == "__main__":
    main()