from langchain.schema import HumanMessage, SystemMessage
import os
import requests
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import pandas as pd
import streamlit as st
import plotly.express as px
import statsmodels.api as sm
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from sklearn.cluster import KMeans
import warnings

warnings.filterwarnings("ignore", message="is_sparse is deprecated")

# Function to display images (header/footer) with a local fallback
def display_image(local_path, fallback_url):
    if os.path.exists(local_path):
        st.image(local_path, use_container_width=True)
    else:
        # Check if the fallback URL is valid before displaying
        try:
            response = requests.head(fallback_url)
            if response.status_code == 200:
                st.image(fallback_url, use_container_width=True)
            else:
                st.error(f"Image not found at: {fallback_url}")
        except requests.RequestException:
            st.error(f"Failed to load image from: {fallback_url}")

# Helper function to make column labels more professional
def professional_label(col_name: str) -> str:
    """Remove 'per 100g' from nutrient names and rename 'Calories per 100g' to 'Calories'."""
    if col_name == "Calories per 100g":
        return "Calories"
    return col_name.replace(" per 100g", "")

# Helper function to assign clusters with custom ordering based on average calories
def assign_clusters(df, feature, calories_col, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df[[feature, calories_col]])
    df["cluster"] = clusters
    # Compute mean calories per cluster and sort descending
    cluster_means = df.groupby("cluster")[calories_col].mean().sort_values(ascending=False)
    mapping = {old_label: new_label for new_label, old_label in enumerate(cluster_means.index)}
    df["cluster"] = df["cluster"].map(mapping)
    df["cluster"] = df["cluster"].astype(str)  # Convert to string for Plotly
    return df

# Header Image
header_path = "header.png"
header_fallback_url = "https://raw.githubusercontent.com/yildiramdsa/nutrient_composition_of_common_foods_in_canada_analyzing_the_canadian_nutrient_file/main/csv_chatbot/header.png"
display_image(header_path, header_fallback_url)

# Streamlit UI
st.title("What’s in Your Food? A Data-Driven Nutrient Analysis")

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
    allow_dangerous_code=True
)

# Nutrient Analysis & Visualization Section
st.subheader("Discover Your Food’s Nutrient Profile")

# Define nutrient columns (exclude non-nutrient columns and "Calories per 100g")
if not df.empty:
    nutrient_columns = [
        col for col in df.columns 
        if col not in ['Food Name', 'Food Subcategory', 'Food Category', 'Calories per 100g']
    ]
else:
    nutrient_columns = []

selected_nutrient = st.selectbox(
    "Select Nutrient",
    ["None Selected"] + sorted(nutrient_columns),
    key="nutrient_select"
)

if selected_nutrient != "None Selected":
    title_nutrient = professional_label(selected_nutrient)
    
    # Bar Chart: Average nutrient by Food Category
    grouped_category = (
        df.groupby("Food Category", as_index=False)[selected_nutrient]
        .mean()
        .sort_values(selected_nutrient, ascending=False)
    )
    fig = px.bar(
        grouped_category,
        y="Food Category", 
        x=selected_nutrient,
        title=f"Average {title_nutrient} by Food Category",
        orientation='h',
        color_discrete_sequence=["#f6a600"]
    )
    fig.update_traces(texttemplate='%{x:.2f}', textposition='outside')
    st.plotly_chart(fig)
    
    # Scatter Plot: Nutrient vs Calories by Food Category (Clusters)
    grouped_df = (
        df.groupby("Food Category", as_index=False)
        .agg({"Calories per 100g": "mean", selected_nutrient: "mean"})
        .dropna()
    )
    grouped_df = assign_clusters(grouped_df, selected_nutrient, "Calories per 100g", 3)
    fig_cal = px.scatter(
        grouped_df,
        x=selected_nutrient,
        y="Calories per 100g",
        title=f"Food Category Clusters - {title_nutrient} vs Calories",
        color="cluster",
        labels={selected_nutrient: title_nutrient, "Calories per 100g": "Calories"}
    )
    st.plotly_chart(fig_cal)

# -------------------------------
# Chatbot Section
# -------------------------------
st.subheader("Ask About Your Food’s Nutrition")

if "chatbot_answer" not in st.session_state:
    st.session_state["chatbot_answer"] = None

question = st.text_area("Enter your question:", "")

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

# Footer Image
footer_path = "footer.png"
footer_fallback_url = "https://raw.githubusercontent.com/yildiramdsa/nutrient_composition_of_common_foods_in_canada_analyzing_the_canadian_nutrient_file/main/csv_chatbot/footer.png"
display_image(footer_path, footer_fallback_url)
