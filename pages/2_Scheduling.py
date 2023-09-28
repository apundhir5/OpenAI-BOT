import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.llms import OpenAI
import openai
import speech_to_text
import speech_recognition as sr

load_dotenv()
openai.api_key_path = None
Open_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = 'gpt-3.5-turbo'

@st.cache_data
def get_data():
    columnnames = ['StartTime','EndTime','ZipCode','EmployeeName','lat','lon']
    df = pd.read_csv('./Data/schedule.csv', dtype={
                                            'StartTime': 'string',
                                            'EndTime': 'string',
                                            'ZipCode': 'string',
                                            'EmployeeName': 'string',
                                            'lat': 'float64',
                                            'lon': 'float64'
        })
    return df

import re
def find_zip_codes(s):
    #Search for all ZIP codes in a given string.
    pattern = r'\b\d{5}(?:-\d{4})?\b'
    return re.findall(pattern, s)


def make_prompt(question: str, context: str, data: str):
    prompt = (
            "You are an assistant that can crew scheduling and is expert Data Analyst.\n"
            "Answer the following question using the provided context.\n"
            "Do not include any information in your response that is not in the provided context.\n"
            "Use the data provided after Data.\n"
            "Do not mention the existence of the context to the user.\n"
            "Keep your answer concise and be precise with numbers; Use 3 sentences at most.\n"
            "Context:\n"
            f"{context}\n"
            "Data:\n"
            f"{data}\n"
            "Question:\n"
            f"{question}"
        )
    return prompt

def get_answer(question):

    df = get_data()
    prompt = make_prompt(question, get_context(), df.to_string())  

    answer = openai.ChatCompletion.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
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


def main():
    st.title("Speech to Text - Gen AI Scheduling BOT")

    map_container = st.container()

    if 'text_input_value' not in st.session_state:
        st.session_state.text_input_value = ''

    # question = st.text_input('Question:')

    if "Record" not in st.session_state:
        st.session_state["Record"] = False

    if st.button("Record"):
        st.session_state["Record"] = not st.session_state["Record"]
        # st.session_state.text_input_value = 'clicked'
        # create recognizer and mic instances
        recognizer = sr.Recognizer()
        microphone = sr.Microphone()

        audio_data = speech_to_text.recognize_speech_from_mic(recognizer, microphone)
        if (audio_data['success']):
            # question = st.text_input('Question:', value=audio_data['transcription'])
            st.session_state.text_input_value = audio_data['transcription']
        else:
            st.write(audio_data['error'])

    question = st.text_input('Question:', value=st.session_state.text_input_value)

    if "Submit" not in st.session_state:
        st.session_state["Submit"] = False

    # # if st.session_state["button1"]:
    # if st.button("Submit"):
    #     # toggle button3 session state
    #     st.session_state["Submit"] = not st.session_state["Submit"]

    # if st.session_state["Submit"]:
    if st.button("Submit"):
        # st.write("**Submit!!!**")
        with map_container:
            answer = get_answer(question)
            
            # Extract ZIP codes from the provided string
            zip_codes = set(find_zip_codes(answer))
            df = get_data()
            # Add a new column based on the existence of the ZIP code in the provided string
            df['color'] = df['ZipCode'].apply(lambda x: '#0bfc03' if x.strip() in zip_codes else '#fc032d')
    
            st.write(answer)
            st.map(df, color='color')


# def main_OLD():
#     st.title("PGE Gen AI Scheduling BOT")

#     map_container = st.container()

#     # Processing only occurs after the user clicks "submit"
#     with st.form("scheduling_form", clear_on_submit=True):
#         # Check if the text area value is in the session state
#         # if 'text_area_value' not in st.session_state:
#         #     st.session_state.text_area_value = ''

#         if 'text_input_value' not in st.session_state:
#             st.session_state.text_input_value = ''

#           # Create a text area with the session state value
#         # response = st.text_area('Response:', value=st.session_state.text_area_value, height=300)
    
#         # Ask a question and get input in a text box
#         question = st.text_input('Question:')

#         submitted = st.form_submit_button("Submit")

#     # If the user submits the form
#     if submitted:
#         with map_container:
#             st.write(get_answer(question))
#             st.map(get_data())

#         # st.experimental_rerun()
        
if __name__ == "__main__":
    main()
