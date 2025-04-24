
import numpy as np
from transformers import (AutoTokenizer, AutoModel)
import time
from PyPDF2 import PdfReader
import os
import re
import config
VERBOSE=config.VERBOSE

def clean_paragraph(p):
    # remplace tous les séparateurs de ligne Unicode possibles par un espace
    p = re.sub(r'[\r\n\x0b\x0c\u2028\u2029]+', ' ', p)
    return p.strip()

import tkinter as tk
from tkinter import filedialog

def select_directory():
    root = tk.Tk()
    root.withdraw()  # cache la fenêtre principale
    folder_path = filedialog.askdirectory(title="Select a folder")
    return folder_path
def clean_paragraph(text: str) -> str:
        return ' '.join(text.split())  # remove extra whitespace/newlines



class RAGDataset:
    def __init__(self,load:bool,data_path:str=None,chunk_size=config.CHUNK_MAX_SIZE):
        assert os.path.exists(data_path),f"incorrect path to data : {data_path}"
        assert chunk_size.is_integer, f"need chunk_size as integer"
        assert chunk_size>=10, f"need chunk_size >=10"
        self.work_path=data_path
        self.context_path = os.path.join(self.work_path,"context.txt")
        self.refined_path= os.path.join(self.work_path,"refined.txt")
        self.meta_path= os.path.join(self.work_path,"meta.txt")
        pdf_paths = []
        if not load:
            with open(self.context_path, "w", encoding="utf-8") as f:
                pass
            with open(self.refined_path, "w", encoding="utf-8") as f:
                pass
            with open(self.meta_path, "w", encoding="utf-8") as f:
                pass
            for nom_fichier in os.listdir(data_path):
                chemin_complet = os.path.join(data_path, nom_fichier)
                if os.path.isfile(chemin_complet) and nom_fichier.lower().endswith(".pdf"):
                    pdf_paths.append(chemin_complet)
            for path in pdf_paths:
                self.extractPDF(path,self.context_path,self.meta_path)
            self.refineTXT(self.context_path, self.refined_path)
        
        

        self.context_lines=open(self.context_path, 'r', encoding="utf-8").readlines()
        self.refined_lines=open(self.refined_path, 'r', encoding="utf-8").readlines()
        
        
        if len(pdf_paths)>0: 
            meta=open(self.meta_path, 'r', encoding="utf-8").readlines()
            self.meta_lines=[]
            for ligne in meta:
                self.meta_lines.append(ligne.split('\t'))
            self.dataset=self.make_context("pdf",meta=True)
        else:
            self.dataset=self.make_context("folder",meta=False)



    

    def extractPDF(self,pdf_path: str, txt_output_path: str, meta_output_path: str, chunk_size=config.CHUNK_MAX_SIZE,chunk_overlap=config.CHUNK_OVERLAP):
        file_name = os.path.basename(pdf_path)
        reader = PdfReader(pdf_path)
        print("[DEBUG] init extractPDF")
        chunk = []
        next_chunk = []

        with open(txt_output_path, "w", encoding="utf-8") as f_txt, \
            open(meta_output_path, "w", encoding="utf-8") as f_meta:

            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text:
                    continue  # skip empty pages

                words = text.split()  # split by word

                for word in words:
                    chunk.append(word)

                    # If we're nearing the chunk size, start saving overlap words
                    if chunk_size - len(chunk) <= chunk_overlap:
                        next_chunk.append(word)

                    # When chunk is full, write it and prepare next
                    if len(chunk) >= chunk_size:
                        paragraph = clean_paragraph(' '.join(chunk))
                        assert len(paragraph.split(" "))<=chunk_size,"incorrect chunk size"
                        f_txt.write(paragraph + "\n")
                        f_meta.write(f"{page_num + 1}\t{file_name}\n")
                        
                        # Set current chunk to the overlapping words
                        chunk = next_chunk
                        next_chunk = []

            # Write the last remaining chunk if it's not empty
            if chunk:
                paragraph = clean_paragraph(' '.join(chunk))
                assert len(paragraph.split(" "))<=chunk_size,"incorrect chunk size"
                f_txt.write(paragraph + "\n")
                f_meta.write(f"{page_num + 1}\t{file_name}\n")
        print("[DEBUG] end extractPDF")

        


    def refineTXT(self, input_path, output_path):
        with open(input_path, "r", encoding="utf-8") as f_in, \
             open(output_path, "a", encoding="utf-8") as f_out:
            for i in f_in:
                f_out.write(refine(i, filtre) + "\n")

    def make_context(self,type:str,meta=True):
        embeddings=[]
        for i in range(len(self.context_lines)):
            dict_meta={}
            if meta:
                dict_meta={
                    "page": self.meta_lines[i][0],
                    "file": self.meta_lines[i][1]
                }
            embeddings.append({
                "description": self.refined_lines[i],
                "data": self.context_lines[i],
                "metadata": dict_meta
            })
            
        dataset = {"type":type,"embeddings":embeddings,"index":None}
        return dataset
class KnowledgeBase:
    
    """

    """
    def __init__(self,input_rag_dataset:RAGDataset,token_embed_str:str,model_embed_str:str):
        
        start = time.time()
        self.index_path=os.path.join(input_rag_dataset.work_path,"faiss_index.idx")
        
        if not os.path.exists(self.index_path):
            with open(self.index_path, "x", encoding="utf-8") as f:
                pass
        
        from torch import cuda
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.dataset=input_rag_dataset.dataset
        self.loadTokeniser(token_embed_str,model_embed_str)
        end = time.time()
        print(f"[KnowledgeBase] Temps d'exécution : {end - start:.2f} secondes")
    def load_faiss_index(self):
        from faiss import read_index
        self.index = read_index(self.index_path)
        self.dataset["index"]=self.index
    def loadTokeniser(self,token_embed_str:AutoTokenizer,model_embed_str:AutoModel): #EMBED_MODEL = "BAAI/bge-small-en"
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(repo_id="BAAI/bge-small-en", filename="config.json", local_files_only=True)
            print(f"Model is cached at: {path}")
            self.tokenizer_embed = AutoTokenizer.from_pretrained(token_embed_str, local_files_only=True) # tokenize
            self.model_embed = AutoModel.from_pretrained(model_embed_str, local_files_only=True).to(self.device) # vectorize
        except Exception as e:
            print("Model not found locally:", e)
            self.tokenizer_embed = AutoTokenizer.from_pretrained(token_embed_str, local_files_only=False) # tokenize
            self.model_embed = AutoModel.from_pretrained(model_embed_str, local_files_only=False).to(self.device) # vectorize
        
    def build_faiss_index(self):
        start1 = time.time()
        dimension = config.EMBED_DIM #vecteur de 384 dimensions par défaut pour chaque chunk
        from faiss import IndexFlatIP,write_index
        self.index = IndexFlatIP(dimension)
        embeddings = np.vstack([self.get_embedding(q["description"]) for q in self.dataset["embeddings"]])
        self.index.add(embeddings)
        end1 = time.time()
        
        relative_path = os.path.relpath(self.index_path)
        write_index(self.index, relative_path)
        
        self.dataset["index"]=self.index
        end2 = time.time()
        print(f"[build_faiss_index] Temps d'exécution : {end2 - start1:.2f} secondes, avec {end1 - start1:.2f} secondes pour calculer l'index")


    def get_embedding(self, text):
        inputs = self.tokenizer_embed(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        from torch import no_grad
        with no_grad():
            output = self.model_embed(**inputs)
        embedding = output.last_hidden_state[:, 0, :].cpu().numpy()

        # Normalisation L2
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / norm

        return embedding

    
  

class QueryRewriter:
    def __init__(self,rewrite_model:str):
        pass
    def rewrite(self,query_user:str):
        pass
    def verify(self,query_user:str,query:str):
        pass
    def set_context(self,context:str):
        pass
class QueryExpander:
    """
    Expand Module : used to split a big query in several small ones and verify the subqueries obtained.

    Examples:
        >>> expander=QueryExpander(expander_model='llama3.3')
        >>> expander.expand(query,nb_max_query=5)
        [subquery1,subquery2]
    """
    def __init__(self,expander_model:str):
        self.expander_model=expander_model

    def expand(self,query:str,nb_max_query:int): 
        """
        Split a big query in several small ones.

        Args:
            query (string): The query to split.
            nb_max_query (int): The max number of subqueries returned.

        Returns:
            list: The list of subqueries returned.

        Example:
            >>> expander=QueryExpander(expander_model='llama3.3')
            >>> expander.expand(query,nb_max_query=5)
            [subquery1,subquery2]
        """
        pass
    def verify(self,query:str,queries:list):
        pass
    def setModel(self,expander_model:str):
        self.expander_model=expander_model


class VectorFetcher:
    def __init__(self,knowledge:KnowledgeBase):
        self.knowledge=knowledge

    def retrieve(self,query:str,num_queries=5,date_adjust:bool=True,small_to_big=(1,2)):
        start = time.time()
        query_embedding = self.knowledge.get_embedding(query)
        D, I = self.knowledge.index.search(query_embedding, k=num_queries)


        

        bef, aft = small_to_big
        retrieved_infos = []

        for idx in I[0]:
            context_lines = self.knowledge.dataset["embeddings"][idx:idx+1]
            start_idx = max(idx - bef, 0)
            end_idx = min(idx + aft + 1, len(self.knowledge.dataset["embeddings"]))
            combined_data = concaten([self.knowledge.dataset["embeddings"][i]["data"] for i in range(start_idx, end_idx)])
            retrieved_infos.append({
                "description": self.knowledge.dataset["embeddings"][idx]["description"],
                "data": combined_data,
                "metadata": self.knowledge.dataset["embeddings"][idx]["metadata"]
            })
        if VERBOSE>=2:
            print("question: ", query)
            for i in range(len(retrieved_infos)):
                print(f"context: {retrieved_infos[i]['description']} : score = {D[0][i]:.2f}")

        end = time.time()
        print(f"[VectorFetcher] Temps d'exécution : {end - start:.2f} secondes")
        return retrieved_infos


class ChainManager:
    
    class ChainMySQL():
        def retrieve(self,query:str,bdd:str):
            pass
        
    class ChainSem(): 
        """
        A chain to generate queries to question a endpoint OWL/SPARQL

        Examples:
        >>> chainSPARQL=ChainSem(query_endpoint="http://localhost:3030/cluedo/query",model='llama3')
        >>> answer = chainSPARQL.ask("Combien y a-t-il de pièces dans la maison?")
        >>> answer['result']

        'According to the available information, there are 11 pieces in the house.'

        >>> answer['sparql_query']

        PREFIX lamaisondumeurtre: <http://www.lamaisondumeurtre.fr#>
        SELECT (COUNT(?piece) AS ?count)
        WHERE {
            ?house a lamaisondumeurtre:Maison .
            ?house lamaisondumeurtre:pieceDansMaison ?piece .
        }
        
        """
        
        def __init__(self,query_endpoint:str,model:str):
            
            from langchain_ollama import OllamaLLM
            from langchain_community.graphs import RdfGraph
            from langchain_community.chains.graph_qa.sparql import GraphSparqlQAChain

            self.query_endpoint=query_endpoint
            self.graph = RdfGraph(
                query_endpoint=self.query_endpoint,
                standard="rdf",
                local_copy="test.ttl",
            )
            self.model=model
            llm = OllamaLLM(model=self.model)
            verbose=VERBOSE>=1
            self.chain = GraphSparqlQAChain.from_llm(llm, graph=self.graph, verbose=verbose,allow_dangerous_requests=True, return_sparql_query= True)

        def ask(self,question:str):
            response = self.chain.invoke(question)
            return response



class RAGGenerator:
    def generate(self,query:str,context:str,model:str):
        
        from ollama import chat
        input_text = f"context: {context} question: {query}"

        response = chat(model=model, messages=[
            {
                'role': 'system',
                'content': 'répond à la question en français, n\'invente rien, ne doute jamais du contexte qui t\'est donné, dis clairement si tu ne sais pas la réponse. Cite la page d\'origine des informations essentielles ainsi que le nom du fichier d\'où provient l\'information. '
            },
            {
                'role': 'user',
                'content': input_text,
            },
        ])

        return response.message.content

class UserPrompt:
    def __init__(self,fetcher:VectorFetcher):
        self.fetcher=fetcher
    def ask(self,user_query,nb_contextes,small_to_big=config.SMALL_TO_BIG): #TODO dynamic small_to_big
        start = time.time()
        print("\n\n---------------------------\n",user_query)
        context=self.fetcher.retrieve(user_query,num_queries=nb_contextes,small_to_big=small_to_big)

        str_context=""
        for i in range(nb_contextes):
            str_context+=context[i]["data"]+str(context[i]["metadata"])+" \n"
        print("\n\n---------------------------\n",user_query)
        print(f"[ask] generation in progress using {config.GENERATOR_MODEL} please wait ...")
        generator=RAGGenerator()
        response = generator.generate(query=user_query,context=str_context,model=config.GENERATOR_MODEL)
        if VERBOSE>=1:print(response)
            
        end = time.time()
        if VERBOSE>=1:print(f"[ask] Temps d'exécution : {end - start:.2f} secondes")
        return response
    
    
        return response
    def askloop(self):
        import sys
        if sys.platform == "win32":
            try:
                from pyreadline3.rlmain import Readline # assure le support readline sous Windows
            except ImportError:
                print("Pour une meilleure expérience, installez pyreadline3 avec `pip install pyreadline3`")
        else:
            import readline  # fonctionne nativement sur Unix/Linux/Mac
        global VERBOSE
        nb = config.NB_CONTEXTES

        print("================================================")
        print("Vous pouvez changer le niveau de VERBOSE avec /v")
        print(f"Vous pouvez changer le nombre de contextes récupérés avec (par défaut {config.NB_CONTEXTES})")
        print("Vous pouvez quitter avec q, x, \"\", quit, exit")
        print("================================================")

        while True:
            try:
                user_input = input("Bonjour quelle est votre question ?\n").strip()

                if user_input in ("q", "x", "", "quit", "exit"):
                    break

                
                readline = Readline()
                readline.add_history("ta question")  # Ajoute la question à l'historique

                if user_input == "/v":
                    v = input("Quel niveau de VERBOSE ? \n").strip()
                    while not v.isnumeric() or not (0 <= int(v) <= 6):
                        print("Valeur incorrecte (0 à 6)")
                        v = input("Quel niveau de VERBOSE ?\n").strip()
                    VERBOSE = int(v)

                elif user_input == "/n":
                    nb_str = input("Combien de contextes donner ?\n").strip()
                    while not nb_str.isnumeric() or not (2 <= int(nb_str) <= 20):
                        print("Valeur incorrecte (entre 2 et 20)")
                        nb_str = input("Combien de contextes donner ?\n").strip()
                    nb = int(nb_str)

                else:
                    self.ask(user_input, nb_contextes=nb)

            except KeyboardInterrupt:
                print("\nInterrompu. Tapez 'q' pour quitter.")
            except EOFError:
                print("\nFin de l'entrée. Au revoir.")
                break


    def askfile(self):
        pass


    




def cut(l,param):
    return l[param[1]-param[0]:param[1]+param[2]+1:]
def concaten(l):
    ret=""
    for i in l:
        ret+=i+" "
    return ret
def filtre(string:str):
    for i in string:
        if i.isnumeric():
            return False
        if i.isupper():
            return False
    if len(string)>4:
        return False
    else: return True


def refine(string:str,filtre):
    l=string.split()
    li=[]
    for i in l:
        if not filtre(i.strip()):
            li.append(i.strip())
    return concaten(li)



"""
chainSQL1=ChainManager.ChainSem(query_endpoint="http://localhost:3030/cluedo/query",model='llama3')
chainSQL1.ask("Combien y a-t-il de pièces dans la maison ?") #fuseki-server --update --mem /cluedo
dataset2=[
    {"info": "Les registres du processeur XYZ ont une taille de 68 bits.", "date": "2022-01-01", "isChained":False, "chain":None},
    {"info": "Le processeur XYZ possède 8 cœurs physiques et 16 threads.", "date": "2022-01-01", "isChained":True, "chain":chainSQL1}
    ]

"""


VERBOSE=1

if __name__=="__main__":
    
    VERBOSE=2

    
    from torch import cuda
    device = "cuda" if cuda.is_available() else "cpu"
    print("device : ",device)
    
    
    dataset3=[]
    small_to_big = (1,2)


    path=select_directory()
    start = time.time()
    dataset3=RAGDataset(config.LOAD_INDEX,path,30)
    print(f"[RAGDataset] Temps d'exécution : {time.time() - start:.2f} secondes CHUNK_MAX_SIZE = ",30)


    knowledge = KnowledgeBase(dataset3,config.EMBED_MODEL,config.EMBED_MODEL)


    start_index = time.time()
    
    if config.LOAD_INDEX:
        knowledge.load_faiss_index()
    else:
        knowledge.build_faiss_index()
    end_index = time.time()
    print(f"[index] Temps d'exécution : {end_index - start_index:.2f} secondes LOAD = ",config.LOAD_INDEX)
    fetcher=VectorFetcher(knowledge)
    
    
    

    
    end = time.time()
    print(f"[global init] Temps d'exécution : {end - start:.2f} secondes")



    user_query="INSA"
    nb_contextes=5
    small_to_big=(0,0)

    user=UserPrompt(fetcher)
    user.askloop()




