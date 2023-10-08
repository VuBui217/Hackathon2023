from flask import Flask, render_template, request
import pandas as pd
from langchain.llms import OpenAI
from langchain.agents.agent_types import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.agents import create_pandas_dataframe_agent
from datetime import datetime
import openai
app = Flask(__name__)

# Replace 'YOUR_API_KEY' with your actual OpenAI API key
api_key = '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'
# Initialize the OpenAI API client
openai.api_key = api_key


# Load your event data into a DataFrame
df = pd.read_csv("event_data34.csv")

# Create the Langchain agent
agent = create_pandas_dataframe_agent(
    ChatOpenAI(openai_api_key=api_key, temperature=0, model="gpt-3.5-turbo"),
    df,
    verbose=True,
    agent_type=AgentType.OPENAI_FUNCTIONS,
)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_input = request.form["user_input"]

        response = agent.run(user_input)
        return render_template("index.html", user_input=user_input, response=response)
    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
