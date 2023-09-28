import openai
import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv

st.title("Conversational Gen AI Scheduling BOT")

load_dotenv()
openai.api_key_path = None
Open_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = 'gpt-3.5-turbo'

# @st.cache_data
def get_data():
    # columnnames = ['StartTime','EndTime','ZipCode','EmployeeName','lat','lon']
    df = pd.read_csv('./Data/schedule.csv', dtype={
                                            'StartTime': 'string',
                                            'EndTime': 'string',
                                            'ZipCode': 'string',
                                            'EmployeeName': 'string',
                                            'lat': 'float64',
                                            'lon': 'float64'
        })
    return df

def make_prompt(context: str, data: str):
    prompt = (
            "You are an assistant that can crew scheduling and is expert Data Analyst.\n"
            "Answer the following question using the provided context.\n"
            "Do not include any information in your response that is not in the provided context but you can answer questions from your training.\n"
            "Use the data provided after Data.\n"
            "Do not mention the existence of the context to the user.\n"
            "Keep your answer concise and be precise with numbers; Use 3 sentences at most.\n"
            "Context-\n"
            f"{context}\n"
            "Data-\n"
            f"{data}"
        )
    return prompt

def get_answer():
    messages = [{"role": "assistant", "content": get_assistant_prompt()}]

    for m in st.session_state.messages:
        messages.append({"role": m["role"], "content": m["content"]})
        
    answer = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=messages,
            max_tokens=128,
            temperature=0.0,
            top_p=1.0,
            presence_penalty=0.0,
            frequency_penalty=0.0,
        ).choices[0].message.content
    return answer

def get_context():
    return "I have table for schedule with following fields:\
            StartTime,\
            EndTime,\
            ZipCode,\
            EmployeeName,\
            Longitude,\
            Latitude"

def get_assistant_prompt():
    df = get_data()
    assistant_prompt = make_prompt(get_context(), df.to_string())
    return assistant_prompt

if st.button("Reset"):
    for key in st.session_state.keys():
        del st.session_state[key]

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

map_container = st.container()

if prompt := st.chat_input("Ask me about Scheduling"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = get_answer()
        # message_placeholder.markdown(response + "▌")
        message_placeholder.markdown(response)

    with map_container:
        df = get_data()
        st.map(df)

    st.session_state.messages.append({"role": "assistant", "content": response})
