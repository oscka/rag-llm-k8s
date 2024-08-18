import streamlit as st
import requests
import os
# LLM 서비스의 URL (LoadBalancer IP 사용)
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL")
st.title("RAG-Enhanced LLM Chat Interface")
# 사용자 입력 받기
user_input = st.text_input("Enter your prompt:", "")
if st.button("Generate"):
    if user_input:
        # LLM 서비스에 요청 보내기
        response = requests.post(f"{LLM_SERVICE_URL}/generate",
                                 json={"prompt": user_input})
        if response.status_code == 200:
            generated_text = response.json()["generated_text"]
            st.write("Generated response:")
            st.write(generated_text)
        else:
            st.error(f"Error: {response.status_code}, {response.text}")
    else:
        st.warning("Please enter a prompt.")
st.write("This interface uses the Meta-Llama-3.1-8B-Instruct model.")
