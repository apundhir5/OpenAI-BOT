import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
import streamlit as st
import openai
import time

load_dotenv()
openai.api_key_path = None
Open_API_KEY = os.environ.get("OPENAI_API_KEY")
MODEL_NAME = 'gpt-3.5-turbo'

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

def get_data():
    temp_df = pd.read_csv('./Data/energy.csv')
    return temp_df

def get_answer_data(question):
    df = get_data()
    query = (
            "You are an assistant that can review energy effeciency and is expert Data Analyst.\n"
            "Answer the following question using the provided context.\n"
            "Do not include any information in your response that is not in the provided context.\n"
            "Use the data provided after Data.\n"
            "Do not mention the existence of the context to the user.\n"
            "Keep your answer concise; Use 3 sentences at most.\n"
            "Context:\n"
            f"{get_context()}\n"
            "Data:\n"
            f"{df.to_string()}\n"
            "Question:"\
            f"{question}"
    )
        
    agent = create_pandas_dataframe_agent(OpenAI(temperature=0, model_name = MODEL_NAME), 
                                            df, 
                                            verbose=True)

    answer = agent.run(query)
    return answer

def get_prompt_template(context, data):
    template = (
            "You are an assistant that can review energy effeciency and is expert Data Analyst.\n"
            "Answer the following question using the provided context.\n"
            "Do not include any information in your response that is not in the provided context.\n"
            "Use the data provided after Data.\n"
            "Do not mention the existence of the context to the user.\n"
            "Keep your answer concise and be price with numbers; Use 3 sentences at most.\n"
            "Context:\n"
            f"{context}\n"
            "Data:\n"
            f"{data}\n"
            "Question:"\
            "{question}"
        )
    
    prompt_template_name = PromptTemplate(
        input_variables=['question'],
        template = template
    )
    return prompt_template_name

def get_answer_langchain(question):
    df = get_data()
    prompt_template = get_prompt_template(get_context(), df.to_string())
    
    llm = OpenAI(model_name=MODEL_NAME, temperature=0.6)
    chain = LLMChain(llm=llm, prompt=prompt_template)

    from langchain.chains import SimpleSequentialChain
    chain = SimpleSequentialChain(chains = [chain])
    answer = chain.run(question)
    return answer

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

def make_prompt(question: str, context: str, data: str):
    prompt = (
            "You are an assistant that can review energy effeciency and is expert Data Analyst.\n"
            "Answer the following question using the provided context.\n"
            "Do not include any information in your response that is not in the provided context.\n"
            "Use the data provided after Data.\n"
            "Do not mention the existence of the context to the user.\n"
            "Keep your answer concise and be price with numbers; Use 3 sentences at most.\n"
            "Context:\n"
            f"{context}\n"
            "Data:\n"
            f"{data}\n"
            "Question:\n"
            f"{question}"
        )
    return prompt

def get_context():
    return "I am sharing some details on how energy prices are calculated and also some usage data.\
                Which base rate fits your lifestyle?\
                These are your base rate options for providing electricity to your home. Since everyone's electric needs are different, consider when you use electricity the most and when you can conserve it. The more flexibility you have to shift your energy use to off-peak times, the lower your bill will be.\
                Time of Day Rate - 3 p.m. - 7 p.m. Our new standard base rate favored by most\
                * Shortest time commitment for on-peak pricing of all base rate options.\
                * Best if you can shift usage to before 3 p.m. or after 7 p.m. and all weekend long.\
                * The on-peak rate is lowest October through May.\
                Time of Day Rate - 3 p.m. - 7 p.m. Alternative base rate to save during evenings and weekends\
                * Moderate savings during off-peak hours.\
                * Best if you can shift usage to evenings after 7 p.m. and all weekend long.\
                * The on-peak rate is lowest November through May.\
                * Dynamic Peak Pricing - Alternative base rate for flexible, budget-minded people\
                * Multi-tiered pricing with notable savings during off-peak hours.\
                * Best if you can shift usage to evenings after 11 p.m. and all day on weekends and holidays.\
                * Each rate is the same year-round.\
                I have table for energy consumption by a customer with following fields:\
                Monthly Cost,\
                Monthly Peak Usage in KW,\
                Monthly Off Peak Usage in KW,\
                Weather Temperature Low,\
                Weather Temperature High,\
                Monthly Cost for similar homes,\
                Last Year Cost for the same month"

def main():
    st.title("Energy Gen AI BOT")
    # st.write("Click the 'Start Recording' button to record your speech and convert it to text.")

    # Check if the text area value is in the session state
    if 'text_area_value' not in st.session_state:
        st.session_state.text_area_value = ''

    # Create a text area with the session state value
    response = st.text_area('Response:', value=st.session_state.text_area_value, height=300)
    
    # Ask a question and get input in a text box
    question = st.text_input('Question:')

    option = st.radio("Choose an Option:", ["LangChain", "DataFrameAgent", "ChatCompletion"], horizontal=True)
    
    # Create a button to update the session state value and refresh the page
    if st.button('Ask Question'):
        if option == 'LangChain':
            st.session_state.text_area_value = get_answer_langchain(question)
        elif option == 'DataFrameAgent':
            st.session_state.text_area_value = get_answer_data(question)
        elif option == 'ChatCompletion':
            st.session_state.text_area_value = get_answer(question)

        st.experimental_rerun()

if __name__ == "__main__":
    main()