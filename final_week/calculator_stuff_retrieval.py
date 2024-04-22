import numpy as np
import pandas as pd
import sys
import os
import requests
import fnmatch
import argparse
import base64
import pickle

#langchain imports
from langchain.llms import VertexAI
from langchain import PromptTemplate
from langchain.embeddings import VertexAIEmbeddings
from langchain.docstore.document import Document
from langchain.chains import LLMChain, RetrievalQA, RetrievalQAWithSourcesChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.chains.router import MultiPromptChain


#agent imports
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.tools import BaseTool
from langchain.llms import OpenAI
from langchain import LLMMathChain, SerpAPIWrapper
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL

from typing import Optional, Type

from pydantic import BaseModel, Field

from langchain.callbacks.manager import CallbackManagerForToolRun
from langchain.tools.base import BaseTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)
from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.tools.file_management.write import WriteFileInput
from langchain.agents.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory
from langchain.document_loaders import WebBaseLoader


from langchain.agents import (
    Tool,
    AgentExecutor,
    LLMSingleActionAgent,
    AgentOutputParser,
)
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re
from lists_queries_python_broken import *
from langchain.output_parsers.regex import RegexParser

os.environ['GOOGLE_APPLICATION_CREDENTIALS']=sys.path[-2]+'\claranet-pt-data-ia-2f1d5d4795e4.json'
GITHUB_TOKEN = "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"


####################### BEGIN PROMPT ENGINEERING ###########################
baseline_chain_template = """
You are an AI Python specialist which can perform multiple tasks, some of them being:
    - Give reccomendations about optimizing and simplifing Python code.
    - Answering questions about a GitHub repository python file. 
    - Create unit test in python.
    - Give tips about security of the python code.
You must answer to the questions with a recommendation and always give some reasoning on about your answer.
Prioritize answering with text, giving reccomendations to the user on how to the change the code, than returning python code as your final answer.
Below is an examples of a question and a possible sequence of steps of actions to correctly provide an answer:


Question = {question}
{context}
"""
######################## END PROMPT ENGINEERING ###########################


def main():
    new_db = FAISS.load_local('Calculator_Broken_Python', VertexAIEmbeddings())

    template_prompt = PromptTemplate(template = baseline_chain_template, 
                                     input_variables=['context', 'question']
                                     )



    qa = RetrievalQA.from_chain_type(llm = VertexAI(temperature=0.5, max_output_tokens=200),
                                     chain_type = "stuff",
                                     retriever = new_db.as_retriever(),
                                     chain_type_kwargs={
                                         "prompt": template_prompt
                                     },
                                     return_source_documents = True)


    
    resultado = qa({"query": 'How can I simplify the conditions and code of the my_first_calculator.py file?'})
    print(resultado['result'])
    print(resultado['source_documents'])

if __name__ == '__main__':
    print('\n\n\033[34m' + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + '\034[m')
    print('\n\n\033[34m' + '\nNEW START -- RETRIEVAL\n' + '\033[m')
    print('\n\n\033[34m' + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + '\034[m')

    main()