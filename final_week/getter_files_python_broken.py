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

from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    Language,
)

class CustomTextSplitter:
    def __init__(self, separators):
        self.separators = separators

    def split(self, text):
        split_text = [text]
        for separator in self.separators:
            split_text = [x.split(separator) for x in split_text]
            split_text = [item for sublist in split_text for item in sublist]  # flatten list
        return split_text

os.environ['GOOGLE_APPLICATION_CREDENTIALS']=sys.path[-2]+'\claranet-pt-data-ia-2f1d5d4795e4.json'
GITHUB_TOKEN = "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"



def parse_github_url(url):
    parts = url.strip("/").split("/")
    owner = parts[-2]
    repo = parts[-1]
    return owner, repo


#change the url if wanted
def get_folders_from_github_repo(token):
    url = f'https://api.github.com/repos/edurochasi/python-code-disasters' #repo url
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        content = response.json()
        content = content["contents_url"][0:-8] #removes the /{+path} from the link
        folders = requests.get(content, headers=headers).json()
        return folders #returns a list of dicts with every folder and file of the repo
    else:
        raise ValueError(f"Error fetching repo contents: {response.status_code}")


#function to change when wanted different files
def get_all_files_recursive(folders, token, unalista, boolean):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    while boolean:
        for folder in folders:
            print('folder type: ', folder['type'])
            if fnmatch.fnmatch(folder["path"], "*.jpg") or fnmatch.fnmatch(folder["path"], "*.txt") or fnmatch.fnmatch(folder["path"], "*.png"):
                continue
            if folder["type"] == "dir" or folder["type"] == "tree" and folder['path'] != 'libs/langchain/tests':
                folder_folder = requests.get(folder["url"], headers=headers)
                files = folder_folder.json()
                get_all_files_recursive(files, token, unalista, boolean)
            if folder["type"] == "blob" or folder["type"] == "file":
                unalista = add_files_to_list(folder, unalista, GITHUB_TOKEN)


         
        
        return unalista
    return unalista

#no need to change
def add_files_to_list(file, dulista, token):
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github+json"
    }
    #if "git_url" in file.keys():
    #    response = requests.get(file["git_url"], headers = headers)
    #else:
    response = requests.get(file["url"], headers = headers)
    
        
    content = response.json()["content"]
    decoded = base64.b64decode(content).decode('utf-8')
    print("Fetching file from: ", file["path"])
    dulista.append(Document(page_content=decoded, metadata={"source": file['path']}))
    return dulista


############## CHOOSE CHUNKING OPTION #############
def get_source_chunks(files):
    print("In get_source_chunks ...")
    source_chunks = []
    python_spliter = RecursiveCharacterTextSplitter.from_language(language=Language.PYTHON, chunk_size = 8192, chunk_overlap = 100)
    #splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=100)
    for source in files:
        print("chunking from: ", source.metadata)
        for chunk in python_spliter.split_text(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    return source_chunks

def get_source_chunks_custom_separators(files):
    print("In get_source_chunks ...")
    source_chunks = []
    custom_splitter = CustomTextSplitter(["\nclass ", "\ndef ", "\n\tdef "])
    
    for source in files:
        print("chunking from: ", source.metadata)
        for chunk in custom_splitter.split(source.page_content):
            source_chunks.append(Document(page_content=chunk, metadata=source.metadata))
    return source_chunks


#automated function from main()
def initialization(link, token, boolean, state_of_print = False):
    GITHUB_OWNER, GITHUB_REPO = parse_github_url(link)
    print(GITHUB_OWNER, GITHUB_REPO)
    all_folders = get_folders_from_github_repo(token)

    lista_files = []
    all_files = get_all_files_recursive(all_folders, GITHUB_TOKEN, lista_files, boolean)
    print(f'FETCHED COMPLETED: SUCCESS!!!!!!!! You have fetched {len(all_files)} files')
    if state_of_print: #does not print by default
        print(all_files)
    return all_files


def main():


    link = "https://github.com/edurochasi/python-code-disasters"
    boolean = True

    #if necessary to 'eat' the files of the Repo
    #change url to api github in get_folders_from_github()
    all_files = initialization(link, GITHUB_TOKEN, boolean)

    ##### FAISS ########
    source_chunks = get_source_chunks(all_files)
    new_db = FAISS.from_documents(source_chunks, VertexAIEmbeddings())
    new_db.save_local('ALL_Broken_Python_8192')
    print('db saved')


    #new_db = FAISS.load_local('first_fase_ReScope', VertexAIEmbeddings()) 
    #print('db loaded')



if __name__ == "__main__":
    print('\n\n\033[36m' + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + '\033[m')
    print('\n\n\033[36m' + '\nGETTER_FILES_PYTHON_BROKEN\n' + '\033[m')
    print('\n\n\033[36m' + '$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$' + '\033[m')

    main()
