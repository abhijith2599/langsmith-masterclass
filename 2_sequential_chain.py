import os
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# to set a different project name for langsmith overriding the one on .env
os.environ["LANGCHAIN_PROJECT"]="sequentail llm run"
# Now we will see this code run as a different project while others use the project name in .env

prompt1 = PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Generate a 5 pointer summary from the following text \n {text}',
    input_variables=['text']
)

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY is not set")

model = ChatGoogleGenerativeAI(api_key=API_KEY, model="gemini-2.5-flash", temperature=0.7)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

# If we want to pass some ourown metadate for tracing or other things we can do that by setting up a config and then pass that config to the chain.invoke
config = {
    'run_name':"sequentail run", #to chgange the run or trace name
    'metadata': {
        'subject':"Unemployment in India", #to set some tags
    }
}

result = chain.invoke({'topic': 'Unemployment in India'}, config=config)

print(result)
