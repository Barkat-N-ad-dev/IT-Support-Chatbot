import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image

# Load and cache model/tokenizer
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    model = AutoModelForCausalLM.from_pretrained(
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto"
    )
    return tokenizer, model

# Extract ITBot's response cleanly
def get_response(user_question, tokenizer, model):
    prompt = (
        f"<s>[INST] You are ITBot, a helpful and knowledgeable IT Support Assistant. "
        f"Your job is to assist users with technical issues.\n\n"
        f"User: {user_question} [/INST]"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True
    )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    # Try to extract only ITBot's reply
    if "ITBot:" in decoded:
        reply = decoded.split("ITBot:")[-1].strip()
    elif "[/INST]" in decoded:
        reply = decoded.split("[/INST]")[-1].strip()
    else:
        reply = decoded.replace(prompt, "").strip()

    # Remove any repeated instruction text if any
    if "User:" in reply:
        reply = reply.split("User:")[0].strip()

    return reply

# Streamlit setup
st.set_page_config(page_title="IT Support Chatbot", layout="wide")
st.title("🤖 IT Support Chatbot")

# Greet the user
if "greeted" not in st.session_state:
    st.info("👋 Hi! I'm **ITBot**, your IT support assistant. Ask me anything about your computer, software, internet, or errors!")
    st.session_state["greeted"] = True

# Screenshot upload
uploaded_image = st.file_uploader("📎 Upload a screenshot of the issue (optional)", type=["png", "jpg", "jpeg"])
if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="📷 Uploaded Screenshot", use_column_width=True)

# Get user's IT question
user_input = st.text_input("💬 What’s your IT issue?")

# Load model/tokenizer
tokenizer, model = load_model()

# Get and display response
if user_input:
    with st.spinner("🤔 ITBot is thinking..."):
        response = get_response(user_input, tokenizer, model)
    st.success("✅ ITBot's Response:")
    st.write(response)
