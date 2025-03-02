import json
import openai
import os
import pandas as pd
import plotly.express as px
import panel as pn
from google.colab import drive
import autogen
from autogen import AssistantAgent, GroupChat, GroupChatManager

# Load API Key
config_list = autogen.config_list_from_json("/content/OPENAI_API_KEY.json")
model_name = config_list[0].get("model")
api_key = config_list[0].get("api_key")

if model_name is None or api_key is None:
    raise ValueError("The required keys are missing in the configuration.")
else:
    print(f"Using Model: {model_name}")

# Upload Dataset
from google.colab import files
uploaded = files.upload()
file_path = list(uploaded.keys())[0]

# Load Dataset Function
def load_data(file_path):
    if file_path.endswith(".csv"):
        return pd.read_csv(file_path)
    elif file_path.endswith(".json"):
        return pd.read_json(file_path)
    else:
        raise ValueError("Unsupported file format. Please upload a CSV or JSON file.")

df = load_data(file_path)
print("Data Loaded Successfully!")
print(df.head())

# User Proxy Agent (Interface)
# user_proxy = autogen.UserProxyAgent(
#     name="UserProxy",
#     human_input_mode="NEVER",
#     max_consecutive_auto_reply=5,
#     description="Acts as the interface for the user, relaying commands and managing responses from agents.
# )

# csv file path


# Planner Agent
planner = autogen.AssistantAgent(
    name="Planner",
    llm_config={"config_list": config_list},
    system_message="""
    You are responsible for planning data analysis tasks.
    - Break down the user’s request into smaller tasks.
    - Assign tasks to the correct agents.
    - Ensure the correct sequence of execution.
    - Adjust the plan dynamically if errors occur.
    """
)

# Data Agent
data_agent = autogen.AssistantAgent(
    name="DataAgent",
    llm_config={"config_list": config_list},
    system_message="""
    You handle data loading and cleaning. by writting python code
    - Show all  missing values and duplicates when agents work.
    - Remove all missing and duplicates.
    - Format column names properly.
    - Convert categorical data where needed.
    - Provide a cleaned dataset.
    (Don't use streamlit show all direct)
    """
)

#Column Analyzer
column_analyzer = autogen.AssistantAgent(
    name="ColumnAnalyzer",
    llm_config={"config_list": config_list},
    system_message="""
    You analyze the dataset’s columns through writting python code.
    - Identify numeric, categorical, and text-based columns.
    - Detect missing values and suggest fixes.
    - Identify relationships between variables.
    - Check for outliers.
    - Show step by step visualization
    """
)

# Data Summary Agent (Generates Insights)
data_summary = autogen.AssistantAgent(
    name="DataSummary",
    llm_config={"config_list": config_list},
    system_message="""
    You generate summary insights of the dataset.
    - Calculate mean, median, and all summary of the data
    - Identify trends and correlations.
    - Highlight anomalies.
    - Recommend useful visualizations.

    """
)

#Code Writer (Generates Python Code)
code_writer = autogen.AssistantAgent(
    name="CodeWriter",
    llm_config={"config_list": config_list},
    system_message="""
    You generate Python code for data analysis and visualization and show all dashboard material.
    - Write efficient, well-commented code don't use streamlit
    - Use Plotly,Matplotlib,Dash for interactive visualizations.
    - Make Dashboard adjust chat and graph according to data
    - Ensure column validation before plotting.
    - Include error handling in generated code.
    - provide the link of local host
    """
)

#Code Executor (Runs Code)
code_executor = autogen.AssistantAgent(
    name="CodeExecutor",
    llm_config={"config_list": config_list},
    system_message="""
    You execute Python code and return results.
    - Run the code safely.
    - Catch and report errors.
    - Display visualizations properly.
    """
)

# Debugger
debugger = autogen.AssistantAgent(
    name="Debugger",
    llm_config={"config_list": config_list},
    system_message="""
    You debug errors in the generated code.
    - Identify syntax and logical errors.
    - Fix the code while maintaining its purpose.
    - Ensure best practices are followed.
    """
)

# Critique Agent
critique_agent = autogen.AssistantAgent(
    name="CritiqueAgent",
    llm_config={"config_list": config_list},
    system_message="""
    You review outputs for accuracy.
    - Ensure correct data representation.
    - Identify misleading trends.
    - Suggest improvements.
    """
)
def custom_speaker_selection(conversation_state):

# Group Chat
group_chat = autogen.GroupChat(
    agents=[
         planner, data_agent, column_analyzer,
        data_summary, code_writer, code_executor, debugger, critique_agent
    ],
    messages=[],
    max_round=20,
    # speaker_selection_method=state_transition,
)

# Group Manager (Manages the Execution Flow)
group_manager = autogen.GroupChatManager(
    groupchat=group_chat,
     llm_config={"config_list": config_list}
)
task_prompt = """Please help me to build a Dashboard of given data .
- Adjust all the graph chart size according to data.
"""

# Execute Workflow
# def execute_workflow(last_speaker,group_chat):
#     # """
#     # Runs the full agentic workflow based on the user's request.
#     # """
#     # print(" User Query:", user_query)
#     message=groupchat.messages
#     #
#     if last_speaker is planner:
#       return data_agent
#     elif last_speaker in[data_agent,data_summary,column_analyzer]:
#       return code_executor
#     elif last_speaker is code_executor:
chat_result=planner.initiate_chat(group_manager,message=task_prompt)


