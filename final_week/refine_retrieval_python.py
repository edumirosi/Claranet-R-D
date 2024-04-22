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
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.chains.router import MultiPromptChain
from langchain.chains import RetrievalQA


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
We have the opportunity to refine the given answer with some additional context: {context_str}
Given the new context, refine the original answer to better answer the question. If the context isn't useful, return the original answer.

You are an AI Python specialist which can perform multiple tasks, some of them being:
    - Give reccomendations about optimizing and simplifing Python code.
    - Answering questions about a GitHub repository python file. 
    - Create unit test in python.
    - Give tips about security of the python code.
You must answer to the questions with a recommendation and always give some reasoning on about your answer.
Prioritize answering with text, giving reccomendations to the user on how to the change the code, than returning python code as your final answer.

Answer the question: {question}
"""
######################## END PROMPT ENGINEERING ###########################


refine_refine = """
The original question is as follows: {question}
We have provided an existing answer: {existing_answer}
We have the opportunity to refine the existing answer (only if needed) with some more context below.
{context_str}
Given the new context, refine the original answer to better answer the question. 
If the context isn't useful, return the original answer.
You are an AI Python specialist which can perform multiple tasks, some of them being:
    - Give reccomendations about optimizing and simplifing Python code.
    - Answering questions about a GitHub repository python file. 
    - Create unit test in python.
    - Give tips about security of the python code.
You must answer to the questions with a recommendation and always give some reasoning on about your answer.
Prioritize answering with text, giving reccomendations to the user on how to the change the code, than returning python code as your final answer.

"""

def main():
    new_db = FAISS.load_local('ALL_Broken_Python', VertexAIEmbeddings()) #using the defalut python get_chunks, divides the files into different Documents
    new_db_custom = FAISS.load_local('ALL_Broken_Python_CUSTOM', VertexAIEmbeddings()) #using the get_chunks_custom function, chunks entire files as documents for the db 
    new_db_large = FAISS.load_local('ALL_Broken_Python_8192', VertexAIEmbeddings()) #using the defalut python get_chunks but with bigger chunk size (8192), divides the files into different Documents
    


    question_prompt = PromptTemplate(input_variables=['question', 'context_str'],
                                   template=baseline_chain_template)
    
    refine_prompt = PromptTemplate(input_variables=['question', 'existing_answer', 'context_str'], template=refine_refine)

    qa = RetrievalQA.from_chain_type(llm = VertexAI(temperature=0.5, max_output_tokens=200),
                                     chain_type = "refine",
                                     retriever = new_db.as_retriever(),
                                     chain_type_kwargs={
                                         "question_prompt": question_prompt,
                                         "refine_prompt" : refine_prompt
                                     },
                                     return_source_documents = True)



    qa_custom = RetrievalQA.from_chain_type(llm = VertexAI(temperature=0.5, max_output_tokens=200),
                                     chain_type = "refine",
                                     retriever = new_db_custom.as_retriever(),
                                     chain_type_kwargs={
                                         "question_prompt": question_prompt,
                                         "refine_prompt" : refine_prompt
                                     },
                                     return_source_documents = True)

    qa_large = RetrievalQA.from_chain_type(llm = VertexAI(temperature=0.5, max_output_tokens=200),
                                     chain_type = "refine",
                                     retriever = new_db_large.as_retriever(),
                                     chain_type_kwargs={
                                         "question_prompt": question_prompt,
                                         "refine_prompt" : refine_prompt
                                     },
                                     return_source_documents = True)


    #$$$$$$$$$$$$$$$$$$$$ QUERIES $$$$$$$$$$$$$$$$$$$$$$$#

    queries_on = True

    while queries_on:
        #states of queries
        quality_analysis = False
        security_analysis = True
        performance_optimization = False
        code_refactoring = False
        dependency_analysis = False
        code_review = False
        documentation = False
        unit_testing = False


        #code quality analysis
        if quality_analysis:
            for query in list_quality_analysis:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                resultado = qa({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_custom({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
              
        #code security analysis
        if security_analysis:
            for query in list_security_analysis:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                resultado = qa({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_custom({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM2 #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_large({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])

        #code performance optimization
        if performance_optimization:
            for query in list_performance_optimization:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                resultado = qa({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_custom({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])

        
        #code refactoring
        if code_refactoring:
            for query in list_code_refactoring:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                resultado = qa({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_custom({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
        #code dependecy analysis
        if dependency_analysis:
            for query in list_dependency_analysis:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                resultado = qa({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_custom({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
        #code reviewm
        if code_review:
            for query in list_code_review:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                resultado = qa({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_custom({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])

        #code documentation
        if documentation:
            for query in list_documentation:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                resultado = qa({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_custom({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
        #code unit testing
        if unit_testing:
            for query in list_unit_testing:
                print('\n\n\033[31m' + 'QUERY: ' + '\033[m', query)
                resultado = qa({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
                print("\n\n\033[32m"+"#$#$#$#$#$ CUSTOM #$#$#$#$##$#$#"+"\033[m\n\n")
                resultado = qa_custom({"query": query})
                print(resultado['result'])
                print(resultado['source_documents'])
        #end of queries
        queries_on = False




if __name__ == '__main__':
    print('\n\n\033[34m' + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + '\034[m')
    print('\n\n\033[34m' + '\nNEW START\n' + '\033[m')
    print('\n\n\033[34m' + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + '\034[m')

    main()