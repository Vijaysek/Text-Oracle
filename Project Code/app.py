import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from deep_translator import GoogleTranslator
from gtts import gTTS
import tempfile
import numpy as np

try:
    import speech_recognition as sr
    from speech_recognition import Microphone, Recognizer
    speech_available = True
except ImportError:
    speech_available = False
    sr = None

# Load GPT-2 Model & Tokenizer
@st.cache_resource
def load_resources():
    model_name = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    return model, tokenizer

model, tokenizer = load_resources()

# NADO Auxiliary Model (Simulating Token-Level Guidance)
class NADOModel:
    def __init__(self, guidance_words=None):  # Correct constructor
        self.guidance_words = guidance_words if guidance_words else set()
    
    def modify_probabilities(self, token_probs, generated_tokens, tokenizer):
        word_ids = [tokenizer.encode(w, add_prefix_space=True)[0] for w in self.guidance_words if w]
        for i, token_id in enumerate(word_ids):
            if token_id in token_probs:
                token_probs[token_id] *= 2  # Boost probability for required words
        token_probs /= np.sum(token_probs)  # Normalize
        return token_probs

nado = NADOModel(guidance_words={"AI", "technology", "innovation"})  # Example constraints

# Predict Next Word with NADO Control
def predict_next_word(model, tokenizer, text, temperature=0.6, prev_words=set()): # Added temperature and prev_words
    input_ids = tokenizer.encode(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[:, -1, :]  # Extract last word probabilities
        
        logits = logits / temperature # Apply temperature

        # Convert logits to probabilities
        probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy().flatten()
        
        # Apply NADO Model guidance
        if hasattr(nado, "modify_probabilities"):
            probs = nado.modify_probabilities(probs, input_ids[0], tokenizer)

        top_k_indices = np.argsort(probs)[::-1][:10]  # Get indices of top 10
        top_k_words = [tokenizer.decode([idx]).strip() for idx in top_k_indices]
        return top_k_words #Return top words 
    
# Translation function
def translate_text(text, target_language):
    language_mapping = {
        "English": "en", "French": "fr", "Spanish": "es", "German": "de",
        "Chinese (Simplified)": "zh-CN", "Telugu": "te", "Hindi": "hi",
        "Tamil": "ta", "Kannada": "kn", "Malayalam": "ml"
    }
    try:
        return GoogleTranslator(source='auto', target=language_mapping[target_language]).translate(text)
    except Exception as e:
        return f"Translation error: {e}"

# Text-to-Speech function
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        tts.save(temp_file.name)
        return temp_file.name
    except Exception as e:
        st.error(f"TTS Error: {e}")
        return None

# Speech to Text Function
def speech_to_text():
    if not speech_available:
        return "Speech recognition is unavailable. Please install PyAudio and SpeechRecognition."
    try:
        recognizer = Recognizer()
        with Microphone() as source:
            st.info("🎤 Listening... Speak now!")
            audio = recognizer.listen(source, timeout=10)
            return recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return "⚠ Could not understand the audio. Please speak clearly."
    except sr.RequestError as e:
        return f"❌ API unavailable, error: {e}"
    except Exception as e:
        return f"🎙 Microphone error: {e}"

# Streamlit UI
st.title("🔮 Interactive Predictive Text Generator")

# Initialize session state variables
if "generated_text" not in st.session_state:
    st.session_state.generated_text = ""

# Streamlit Sidebar for Language Selection
st.sidebar.title("🌍 Language Selection")
languages = ["English", "French", "Spanish", "German", "Chinese (Simplified)", "Telugu", "Hindi", "Tamil", "Kannada", "Malayalam"]
target_language = st.sidebar.selectbox("Select Output Language", languages, key="language_select")


# Choose Input Method
st.markdown("<h3 style='color:#FFA500;'>Choose Input Method:</h3>", unsafe_allow_html=True)
input_method = st.selectbox("", ("Text Input", "Voice Input"), key="input_method")

user_input = ""  # Initialize user_input here

if input_method == "Text Input":
    st.markdown("<h3 style='color:#4CAF50;'>Enter the Starting Text:</h3>", unsafe_allow_html=True)
    user_input = st.text_area("Start with some text:", st.session_state.generated_text, height=100, key="user_input")

elif input_method == "Voice Input":
    if st.button("🎙 Record Voice", key="voice_input"):
        speech_result = speech_to_text()
        if "error" not in speech_result.lower():
            st.session_state.generated_text = speech_result  # Set generated text
            st.success("✅ Voice input received successfully!")
            st.rerun()  # Rerun immediately after voice input is set

        user_input = st.text_area("🎧 Recognized Speech", st.session_state.generated_text, key="recognized_speech")

# **CORRECT LOGIC FOR UPDATING GENERATED TEXT**
if input_method == "Text Input": #Only update if the input method is text.
    if user_input != st.session_state.generated_text:  # Check after voice input
        st.session_state.generated_text = user_input  # Update if changed

# Generate the next word options if there is some input
if st.session_state.generated_text.strip():
    top_words = predict_next_word(model, tokenizer, st.session_state.generated_text)
    selected_word = st.selectbox("🔽 Choose the next word:", top_words, key="word_select")

    if st.button("✅ Add Word"):
        if selected_word:
            st.session_state.generated_text += " " + selected_word
            st.rerun()

# ... (Display generated text, translation, TTS, and Reset button remain the same)


# **CORRECT LOGIC FOR UPDATING GENERATED TEXT**
if input_method == "Text Input": #Only update if the input method is text.
    if user_input != st.session_state.generated_text:  # Check after voice input
        st.session_state.generated_text = user_input  # 
        if selected_word:
            st.session_state.generated_text += " " + selected_word
            st.rerun()

# Display the generated text
st.subheader("📜 Final Generated Text:")
st.text_area("Generated text:", st.session_state.generated_text, height=120, key="display_text", disabled=True)

# Translate and display translation
if st.session_state.generated_text.strip(): #Translate only if there is generated text.
    translated_text = translate_text(st.session_state.generated_text, target_language)
    st.markdown(f"""
        <div style='background-color:#b8a4c9; padding: 10px; border-radius: 10px;'>
            <h3 style='color: black;'>🌍 Translated Output ({target_language}):</h3>
            <p style='font-size: 18px;'>{translated_text}</p>
        </div><br>
    """, unsafe_allow_html=True)

    # Convert to speech and play audio
    audio_file = text_to_speech(translated_text)
    if audio_file:
        with open(audio_file, "rb") as file:
            st.audio(file.read(), format="audio/mp3")

# Option to Reset
if st.button("🔄 Reset"):
    st.session_state.generated_text = ""
    st.rerun()