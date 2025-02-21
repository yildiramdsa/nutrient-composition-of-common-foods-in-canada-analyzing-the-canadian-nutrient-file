from langchain.schema import HumanMessage, SystemMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import warnings
warnings.filterwarnings("ignore", message="is_sparse is deprecated")

# Load API key
load_dotenv()
openai_key = st.secrets["OPENAI_API_KEY"]
if not openai_key:
    raise ValueError("OpenAI API key is missing. Set 'OPENAI_API_KEY' in your .env file.")

# Initialize LLM model
llm_name = "gpt-4-turbo"
llm_model = ChatOpenAI(api_key=openai_key, model=llm_name)

# Load dataset
data_path = "csv_chatbot/nutrition_facts.csv"
try:
    df = pd.read_csv(data_path).fillna(0)
except FileNotFoundError:
    raise FileNotFoundError(f"Dataset not found at {data_path}. Check the file path.")

# Create chatbot agent
chatbot = create_pandas_dataframe_agent(
    llm=llm_model,
    df=df,
    verbose=True,
    handle_parsing_errors=True,
    allow_dangerous_code=True,
)

# Streamlit UI
st.title("Nutritional Data Explorer and Chatbot")

st.subheader("Explore Food Nutrient Data")
st.write(df.head())

st.subheader("Summary Statistics")
st.write(df.describe())

# Chatbot Query Section with Session State
st.subheader("Ask the Nutrition Chatbot")
if "chatbot_answer" not in st.session_state:
    st.session_state["chatbot_answer"] = None

question = st.text_area(
    "Enter your question:",
    "Which food categories have the highest nutrient density per calorie, particularly for protein, fat, and non-sugar carbohydrates?",
)

if st.button("Get Answer"):
    QUERY = f"""
    Before answering, follow these steps:
    1. Display all column names to understand the data structure.
    2. Handle missing values, convert data types if necessary.
    3. Perform data analysis to derive insights.
    {question}
    - Always attempt at least two different methods to solve the problem.
    - Never fabricate answers; rely solely on the dataset.
    - Explain how you arrived at the answer, specifying column names used.
    """
    try:
        res = chatbot.invoke(QUERY)
        st.session_state["chatbot_answer"] = res.get("output", "No valid response generated.")
    except ValueError as e:
        st.session_state["chatbot_answer"] = "An error occurred: " + str(e)

if st.session_state["chatbot_answer"]:
    st.write("Chatbot Answer:")
    st.markdown(st.session_state["chatbot_answer"])

# Filtering Section (Category and Subcategory)
st.subheader("Filter Food Data by Category and Subcategory")
filtered_df = df.copy()

food_categories = df["Food Category"].dropna().unique().tolist()
selected_category = st.selectbox("Select Food Category", ["All"] + sorted(food_categories))

if selected_category != "All":
    filtered_df = filtered_df.loc[filtered_df["Food Category"] == selected_category]
    filtered_subcategories = filtered_df["Food Subcategory"].dropna().unique().tolist()
    selected_subcategory = st.selectbox(
        "Select Food Subcategory",
        ["All"] + sorted(filtered_subcategories),
        disabled=False,
    )
else:
    selected_subcategory = st.selectbox(
        "Select Food Subcategory",
        ["Select a category first"],
        disabled=True,
    )

if selected_subcategory != "All" and selected_subcategory != "Select a category first":
    filtered_df = filtered_df.loc[filtered_df["Food Subcategory"] == selected_subcategory]

st.write("Filtered Data:")
st.write(filtered_df)

# Scatter Plot with Trend Line
st.subheader("Nutrient Trends: Scatter Plot with Trend Line")
columns = df.columns[3:].tolist()
x_column = st.selectbox("Select X-axis (Nutrient)", columns)
y_column = st.selectbox("Select Y-axis (Nutrient)", columns)

if st.button("Generate Scatter Plot"):
    if pd.api.types.is_numeric_dtype(filtered_df[x_column]) and pd.api.types.is_numeric_dtype(filtered_df[y_column]):
        # Fit a linear regression model without overwriting the LLM model
        X = sm.add_constant(filtered_df[x_column])
        ols_model = sm.OLS(filtered_df[y_column], X).fit()
        filtered_df["Trend"] = ols_model.predict(X)

        fig = px.scatter(
            filtered_df,
            x=x_column,
            y=y_column,
            hover_data=["Food Name"],
            trendline="ols",
            title=f"Scatter Plot: {x_column} vs. {y_column}"
        )
        st.plotly_chart(fig)
    else:
        st.warning("Please select numeric columns for both X and Y axes.")

# Define nutrient columns (exclude non-nutrient columns)
if not df.empty:
    nutrient_columns = [col for col in df.columns if col not in ['Food Name', 'Food Subcategory', 'Food Category']]
else:
    nutrient_columns = []

st.subheader("Nutrient Analysis")
selected_nutrient = st.selectbox("Select Nutrient", ["Select"] + sorted(nutrient_columns), key="nutrient_select")
filtered_df = df.copy()

if selected_nutrient != "Select":
    # Bar Chart: Average nutrient per Food Category
    fig = px.bar(
        df.groupby("Food Category", as_index=False)[selected_nutrient].mean().sort_values(selected_nutrient, ascending=False),
        y="Food Category", 
        x=selected_nutrient,
        title=f"Average {selected_nutrient} per Food Category",
        labels={"Food Category": "Category", selected_nutrient: f"{selected_nutrient} content"},
        orientation='h'
    )
    st.plotly_chart(fig)
    
    # Scatter Plot: Calories vs Nutrient per Food Category
    grouped_df = df.groupby("Food Category", as_index=False)\
                   .agg({"Calories per 100g": "mean", selected_nutrient: "mean"})\
                   .dropna()
    fig_cal = px.scatter(
        grouped_df,
        x="Calories per 100g",
        y=selected_nutrient,
        color="Food Category",
        title=f"Calories vs {selected_nutrient} per Food Category",
    )
    st.plotly_chart(fig_cal)

    food_categories = df["Food Category"].dropna().unique().tolist()
    selected_category = st.selectbox("Select Food Category", ["All"] + sorted(food_categories), key="category_select")
    
    if selected_category != "All":
        filtered_df = df[df["Food Category"] == selected_category]
        fig = px.bar(
            filtered_df.groupby("Food Subcategory", as_index=False)[selected_nutrient].mean().sort_values(selected_nutrient, ascending=False),
            y="Food Subcategory", 
            x=selected_nutrient,
            title=f"Average {selected_nutrient} per Food Subcategory in {selected_category}",
            labels={"Food Subcategory": "Subcategory", selected_nutrient: f"{selected_nutrient} content"},
            orientation='h'
        )
        st.plotly_chart(fig)
        
        fig_cal = px.scatter(
            filtered_df,
            x="Calories per 100g",
            y=selected_nutrient,
            color="Food Subcategory",
            hover_data=["Food Name"],
            title=f"Calories vs {selected_nutrient} per Food Subcategory",
        )
        st.plotly_chart(fig_cal)
        
        food_subcategories = filtered_df["Food Subcategory"].dropna().unique().tolist()
        selected_subcategory = st.selectbox("Select Food Subcategory", ["All"] + sorted(food_subcategories), key="subcategory_select")
        
        if selected_subcategory != "All":
            final_df = filtered_df[filtered_df["Food Subcategory"] == selected_subcategory]
            final_df = final_df.sort_values(by=selected_nutrient, ascending=False)
            st.write("Filtered Data Table:")
            st.write(final_df[["Food Name", selected_nutrient]])
            
            fig_cal = px.scatter(
                final_df,
                x="Calories per 100g",
                y=selected_nutrient,
                color="Food Name",
                hover_data=["Food Name"],
                title=f"Calories vs {selected_nutrient} per Food Item",
            )
            st.plotly_chart(fig_cal)
