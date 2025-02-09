from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
# import streamlit as st
# from langchain_experimental.agents.agent_toolkits import (
#     create_pandas_dataframe_agent,
# )

# Load environment variables from the .env file.
load_dotenv()

# Retrieve the OpenAI API key from environment variables.
openai_key = os.getenv("OPENAI_API_KEY")

# Initialize the language model with the specified API key and model name.
llm_name = "GPT-4o"
model = ChatOpenAI(api_key=openai_key, model=llm_name)

# Read the credit card default dataset and fill missing values with 0.
df = pd.read_csv("../data/processed/nutrition_facts.csv").fillna(value=0)
# print(df.head()) # We are testing to see if we can extract data from the CSV file.

# # Create a pandas DataFrame chatbot.
# chatbot = create_pandas_dataframe_agent(
#     llm=model,
#     df=df,
#     verbose=True,
#     handle_parsing_errors=True,
# )

# # Define the prompt prefix for querying the chatbot.
# CSV_PROMPT_PREFIX = """
# Before answering the question, please follow these steps:

# 1. Set the pandas display options to show all columns to ensure no data is hidden.
# 2. Retrieve and display the column names of the DataFrame to understand the structure of the data.
# 3. Perform any necessary data preprocessing steps, such as handling missing values, data type conversions, or filtering data.
# 4. Analyze the DataFrame to generate insights and answer the question.

# Ensure to clearly explain your approach, the steps you took, and the column names you used in your analysis. Do not include code snippets in the explanation, just describe the steps.
# """

# # Define the prompt suffix for querying the chatbot.
# CSV_PROMPT_SUFFIX = """
# - **ALWAYS** before giving the final answer, attempt at least two different methods to solve the problem.
#   - Reflect on the results from both methods to determine if they answer the original question accurately.
#   - If the results from the methods differ, try additional methods until you obtain consistent results.
#   - If you are unable to achieve consistent results, state that you are not sure of the answer.
# - If you are confident in the accuracy of the answer, create a comprehensive and well-formatted response using Markdown.
# - **DO NOT FABRICATE AN ANSWER OR USE PRIOR KNOWLEDGE. ONLY USE THE RESULTS FROM YOUR CALCULATIONS**.
# - **ALWAYS** include a detailed explanation of how you arrived at the answer in a section starting with "\n\nExplanation:\n".
#   - In the explanation, mention the specific column names used to derive the final answer. Do not include code snippets in the explanation.
# """

# # Streamlit application for displaying results.
# st.title("Database Chatbots: Interacting with CSV Data")

# st.write("Dataset Preview")
# st.write(df.head())

# # User input for the question.
# question = st.text_input(
#     "Enter your query:",
#     "Which education level has the highest average credit limit?",
# )

# # Run the chatbot and display the result.
# if st.button("Run Query"):
#     QUERY = CSV_PROMPT_PREFIX + question + CSV_PROMPT_SUFFIX
#     try:
#         res = chatbot.invoke(QUERY)
#         final_result = res['output']
#     except ValueError as e:
#         # Check if the error is related to output parsing and handle it.
#         if "output parsing error" in str(e):
#             final_result = "An output parsing error occurred. Please try rephrasing your question."
#         else:
#             final_result = f"An error occurred: {e}"
#     st.write("### Final Answer")
#     st.markdown(final_result)

# # To run the Streamlit app, use the command: streamlit run csv_database_chatbot.py

# # Q:
# # Which education level has the highest average credit limit?
# # What is the highest credit limit?