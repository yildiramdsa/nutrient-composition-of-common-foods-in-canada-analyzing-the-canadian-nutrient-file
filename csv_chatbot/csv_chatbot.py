from langchain.schema import HumanMessage, SystemMessage
import os
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

# Helper function to assign clusters with custom ordering based on average calories
def assign_clusters(df, feature, calories_col, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(df[[feature, calories_col]])
    df["cluster"] = clusters

    # Compute mean calories per cluster and sort descending
    cluster_means = df.groupby("cluster")[calories_col].mean().sort_values(ascending=False)
    mapping = {}
    for new_label, old_label in enumerate(cluster_means.index):
        mapping[old_label] = new_label

    # Remap cluster to ensure 0 -> highest cal, 1 -> middle, 2 -> lowest
    df["cluster"] = df["cluster"].map(mapping)
    # Convert to string so Plotly applies discrete colors, not a continuous scale
    df["cluster"] = df["cluster"].astype(str)
    return df

# Helper function to make column labels more professional
def professional_label(col_name: str) -> str:
    """Remove 'per 100g' from nutrient names and rename 'Calories per 100g' to 'Calories'."""
    if col_name == "Calories per 100g":
        return "Calories"
    # Remove any occurrence of " per 100g"
    return col_name.replace(" per 100g", "")

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

# Define nutrient columns (exclude non-nutrient columns and "Calories per 100g")
if not df.empty:
    nutrient_columns = [
        col for col in df.columns 
        if col not in ['Food Name', 'Food Subcategory', 'Food Category', 'Calories per 100g']
    ]
else:
    nutrient_columns = []

st.subheader("Nutrient Analysis")
selected_nutrient = st.selectbox(
    "Select Nutrient",
    ["None Selected"] + sorted(nutrient_columns),
    key="nutrient_select"
)

if selected_nutrient != "None Selected":
    # Cleaned label for the selected nutrient
    title_nutrient = professional_label(selected_nutrient)
    
    # -------------------------------
    # Bar Chart: Average nutrient by Food Category
    # -------------------------------
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
        labels={
            "Food Category": "Food Category",
            selected_nutrient: title_nutrient
        },
        orientation='h',
        color_discrete_sequence=["#f6a600"]
    )
    fig.update_traces(texttemplate='%{x:.2f}', textposition='outside')
    st.plotly_chart(fig)
    
    # -------------------------------
    # Scatter Plot: Nutrient vs Calories by Food Category (Clusters)
    # -------------------------------
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
        title=f"{title_nutrient} vs {professional_label('Calories per 100g')} by Food Category (Clusters)",
        color="cluster",
        hover_data={
            "Food Category": True,
            selected_nutrient: True,
            "Calories per 100g": True,
            "cluster": False  # Hide cluster ID in hover
        },
        color_discrete_map={
            "0": "#d93b3b",   # highest
            "1": "#f6a600",   # middle
            "2": "#a0d606"    # lowest
        },
        labels={
            selected_nutrient: title_nutrient,
            "Calories per 100g": professional_label("Calories per 100g")
        }
    )
    fig_cal.update_traces(marker=dict(size=12))
    fig_cal.update_layout(showlegend=False)
    st.plotly_chart(fig_cal)

    # -------------------------------
    # Filtering by Food Category
    # -------------------------------
    food_categories = df["Food Category"].dropna().unique().tolist()
    selected_category = st.selectbox(
        "Select Food Category",
        ["None Selected"] + sorted(food_categories),
        key="category_select"
    )
    
    if selected_category != "None Selected":
        filtered_df = df[df["Food Category"] == selected_category]
        
        # Bar Chart: Average nutrient by Food Subcategory
        grouped_subcat = (
            filtered_df.groupby("Food Subcategory", as_index=False)[selected_nutrient]
            .mean()
            .sort_values(selected_nutrient, ascending=False)
        )
        fig = px.bar(
            grouped_subcat,
            y="Food Subcategory", 
            x=selected_nutrient,
            title=f"Average {title_nutrient} by Food Subcategory in {selected_category}",
            labels={
                "Food Subcategory": "Food Subcategory",
                selected_nutrient: title_nutrient
            },
            orientation='h',
            color_discrete_sequence=["#f6a600"]
        )
        fig.update_traces(texttemplate='%{x:.2f}', textposition='outside')
        st.plotly_chart(fig)
        
        # Scatter Plot: Nutrient vs Calories by Food Subcategory (Clusters)
        grouped_subcat_scatter = (
            filtered_df.groupby("Food Subcategory", as_index=False)
            .agg({selected_nutrient: "mean", "Calories per 100g": "mean"})
            .dropna()
        )
        grouped_subcat_scatter = assign_clusters(grouped_subcat_scatter, selected_nutrient, "Calories per 100g", 3)
        fig_cal = px.scatter(
            grouped_subcat_scatter,
            x=selected_nutrient,
            y="Calories per 100g",
            title=f"{title_nutrient} vs {professional_label('Calories per 100g')} by Food Subcategory (Clusters)",
            color="cluster",
            hover_data={
                "Food Subcategory": True,
                selected_nutrient: True,
                "Calories per 100g": True,
                "cluster": False
            },
            color_discrete_map={
                "0": "#d93b3b",
                "1": "#f6a600",
                "2": "#a0d606"
            },
            labels={
                selected_nutrient: title_nutrient,
                "Calories per 100g": professional_label("Calories per 100g")
            }
        )
        fig_cal.update_traces(marker=dict(size=12))
        fig_cal.update_layout(showlegend=False)
        st.plotly_chart(fig_cal)
        
        # Filtering by Food Subcategory
        food_subcategories = filtered_df["Food Subcategory"].dropna().unique().tolist()
        selected_subcategory = st.selectbox(
            "Select Food Subcategory",
            ["None Selected"] + sorted(food_subcategories),
            key="subcategory_select"
        )
        
        if selected_subcategory != "None Selected":
            final_df = filtered_df[filtered_df["Food Subcategory"] == selected_subcategory].copy()
            final_df = final_df.sort_values(by=selected_nutrient, ascending=False)
            st.write("Filtered Data Table:")
            st.write(final_df[["Food Name", selected_nutrient]])
            
            # Scatter Plot: Nutrient vs Calories by Food Item (Clusters)
            final_df = assign_clusters(final_df, selected_nutrient, "Calories per 100g", 3)
            fig_cal = px.scatter(
                final_df,
                x=selected_nutrient,
                y="Calories per 100g",
                title=f"{title_nutrient} vs {professional_label('Calories per 100g')} by Food Item (Clusters)",
                color="cluster",
                hover_data={
                    "Food Name": True,
                    selected_nutrient: True,
                    "Calories per 100g": True,
                    "cluster": False
                },
                color_discrete_map={
                    "0": "#d93b3b",
                    "1": "#f6a600",
                    "2": "#a0d606"
                },
                labels={
                    selected_nutrient: title_nutrient,
                    "Calories per 100g": professional_label("Calories per 100g")
                }
            )
            fig_cal.update_traces(marker=dict(size=12))
            fig_cal.update_layout(showlegend=False)
            st.plotly_chart(fig_cal)
