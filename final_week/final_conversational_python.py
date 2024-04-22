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
from langchain.chains import LLMChain, RetrievalQA, RetrievalQAWithSourcesChain, ConversationalRetrievalChain
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
from langchain.chains.llm import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT, QA_PROMPT
from langchain.chains.question_answering import load_qa_chain

os.environ['GOOGLE_APPLICATION_CREDENTIALS']=sys.path[-2]+'\claranet-pt-data-ia-2f1d5d4795e4.json'
GITHUB_TOKEN = "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"



####################### BEGIN PROMPT ENGINEERING ###########################
question_chain_template = """
You are an AI Python specialist which can perform multiple tasks, some of them being:
    - Give reccomendations about optimizing and simplifing Python code.
    - Answering questions about a GitHub repository python file. 
    - Create unit test in python.
    - Give tips about security of the python code.
You must answer to the questions with a recommendation and always give some reasoning on about your answer.
Prioritize answering with text, giving reccomendations to the user on how to the change the code, than returning python code as your final answer.
Below is an examples of a question and a possible sequence of steps of actions to correctly provide an answer:


Question = {question}
{chat_history}
"""
######################## END PROMPT ENGINEERING ###########################



continuos_chain_template = """
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


def main():
    new_db = FAISS.load_local('ALL_Broken_Python', VertexAIEmbeddings())

    template_prompt = PromptTemplate(template = question_chain_template, 
                                     input_variables=['chat_history', 'question']
                                     )


    llm = VertexAI(temperature=0.5, max_output_tokens=200)

    question_generator = LLMChain(llm=llm, prompt=template_prompt)
    doc_chain = load_qa_chain(llm=llm, chain_type="stuff")
    qa = ConversationalRetrievalChain(
                                     retriever = new_db.as_retriever(),
                                     combine_docs_chain = doc_chain,
                                     question_generator = question_generator,
                                     max_tokens_limit=1024,
                                    )

        
    #$$$$$$$$$$$$$$$$$$$$ QUERIES $$$$$$$$$$$$$$$$$$$$$$$#

    queries_on = True

    while queries_on:
        #states of queries
        quality_analysis = True
        security_analysis = False
        performance_optimization = False
        code_refactoring = False
        dependency_analysis = False
        code_review = False
        documentation = False
        unit_testing = False

        chat_history = []

        #code quality analysis
        if quality_analysis:
            for query in list_quality_analysis:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                print(qa({"question": query, "chat_history": chat_history})['answer'])
            
              
        #code security analysis
        if security_analysis:
            for query in list_security_analysis:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                print(qa({"question": query, "chat_history": chat_history})['answer'])   
        
        #code performance optimization
        if performance_optimization:
            for query in list_performance_optimization:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                print(qa({"question": query, "chat_history": chat_history})['answer'])           

        
        #code refactoring
        if code_refactoring:
            for query in list_code_refactoring:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                print(qa({"question": query, "chat_history": chat_history})['answer'])
        
        #code dependecy analysis
        if dependency_analysis:
            for query in list_dependency_analysis:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                print(qa({"question": query, "chat_history": chat_history})['answer'])

        #code reviewm
        if code_review:
            for query in list_code_review:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                print(qa({"question": query, "chat_history": chat_history})['answer'])


        #code documentation
        if documentation:
            for query in list_documentation:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                print(qa({"question": query, "chat_history": chat_history})['answer'])

        #code unit testing
        if unit_testing:
            for query in list_unit_testing:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                print(qa({"question": query, "chat_history": chat_history})['answer'])

        #end of queries
        queries_on = False




if __name__ == '__main__':
    print('\n\n\033[34m' + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + '\034[m')
    print('\n\n\033[34m' + '\nNEW START -- CONVERSATIONAL\n' + '\033[m')
    print('\n\n\033[34m' + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + '\034[m')

    main() 