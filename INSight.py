import faiss
import ollama
from sentence_transformers import SentenceTransformer
import torch
import numpy as np
from transformers import (AutoTokenizer, AutoModel)
from langchain_ollama import OllamaLLM
import time

class KnowledgeBase:
    
    """

    """
    def __init__(self,dataset:list,token_embed_str:str,model_embed_str:str,index_path:str,load:bool=False,method:int = 1):
        start = time.time()
        self.index_path=index_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loadTokeniser(token_embed_str,model_embed_str)
        self.setDataset(dataset)
        if load: self.index = faiss.read_index(index_path)
        elif method == 1 : 
            self.build_faiss_index()
        elif method == 2 : self.make_index_IP(token_embed_str)
        end = time.time()
        print(f"[KnowledgeBase] Temps d'exécution : {end - start:.2f} secondes")

    def loadTokeniser(self,token_embed_str:AutoTokenizer,model_embed_str:AutoModel):
        self.tokenizer_embed = AutoTokenizer.from_pretrained(token_embed_str) # tokenize
        self.model_embed = AutoModel.from_pretrained(model_embed_str).to(self.device) # vectorize
    def build_faiss_index(self):
        start1 = time.time()
        dimension = 384 #vecteur de 384 dimensions pour chaque token
        self.index = faiss.IndexFlatIP(dimension)
        embeddings = np.vstack([self.get_embedding(q["info"]) for q in self.dataset])
        self.index.add(embeddings)
        end1 = time.time()
        faiss.write_index(self.index,self.index_path) 
        end2 = time.time()
        print(f"[build_faiss_index] Temps d'exécution : {end2 - start1:.2f} secondes, avec {end1 - start1:.2f} secondes pour calculer l'index")

    def make_index_IP(self,token_embed:str):
        start1 = time.time()
        faiss_model = SentenceTransformer(token_embed)
        embeddings = np.array([faiss_model.encode(doc["info"]) for doc in self.dataset], dtype=np.float32)
        # FAISS : Créer un index de recherche (cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner Product = Cosine Similarity si les embeddings sont normalisés
        self.index.add(embeddings)
        end1 = time.time()
        start2 = time.time()
        faiss.write_index(self.index, "faiss_index.idx")
        end2 = time.time()
        print(f"[make_index_IP] Temps d'exécution : {end2 - start1:.2f} secondes, avec {end1 - start1:.2f} secondes pour calculer l'index")

    def setDataset(self,dataset:list):
        self.dataset=dataset
    def addDataset(self,dataset:list):
        self.dataset.append(dataset)

    def get_embedding(self, text):
        inputs = self.tokenizer_embed(text, return_tensors="pt", padding=True, truncation=True).to(self.device)
        with torch.no_grad():
            output = self.model_embed(**inputs)
        return output.last_hidden_state[:, 0, :].cpu().numpy()
    
  

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
    def retrieve(self,query:str,num_queries=5,date_adjust:bool=True): 
        start = time.time()
        query_embedding = self.knowledge.get_embedding(query)
        D, I = self.knowledge.index.search(query_embedding, k=num_queries)
        retrieved_infos = [self.knowledge.dataset[i] for i in I[0]]
        if VERBOSE>=2: 
            print("question: ", query)
            for i in range(0,num_queries):
                print(f"context: {retrieved_infos[i]["info"]} : score = {D[0][i]:.2f}")
        
        end = time.time()
        print(f"[VectorFetcher] Temps d'exécution : {end - start:.2f} secondes")
        return retrieved_infos 
    def verify(self,query:str,queries:list):
        pass
    


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
    def generate(self,query:str,context:str):
        input_text = f"context: {context} question: {query}"

        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'system',
                'content': 'développe ton raisonnement mais n\'invente rien, ne doute jamais du context qui t\'est donné, dis clairement si tu ne sais pas la réponse. Ne fais pas référence au contexte. '
            },
            {
                'role': 'user',
                'content': input_text,
            },
        ])

        return response.message.content

class UserPrompt:
    def __init__(self):
        pass
    def getUserQuery():
        pass
    def getUserFeedback():
        pass


class Rag:
    def __init__(self):
        pass
    def call(self,user_query:str):

        self.rephraser=QueryRewriter()
        self.query=self.rephraser.rewrite(user_query) #query="Quelle matières choisir pour arriver à 25 ects ?"

        self.expander=QueryExpander(expander_model='llama3.3')
        self.queries=self.expander.expand(query,5)

        #toolManager = ToolManager()

        self.fetcher=VectorFetcher()
        self.context=[]

        for query in self.queries:
            retrieved_document=self.fetcher.retrieve(self,query)
            if not retrieved_document.needTool:
                self.context.append(retrieved_document.context)
            else:
                self.toolSQL=ChainManager.ChainMySQL()
                bdd="/path/to/database" #???
                self.context.append(self.toolSQL.retrieve(self,query,bdd).context)
        self.generator=RAGGenerator(generator_model='llama3.3')
        self.generator.generate(query=self.query,context=self.context)




dataset1 = [
            {"info": "Les registres du processeur XYZ ont une taille de 68 bits.", "date": "2022-01-01"},
            {"info": "Les registres du processeur XYZ ont une taille de 36 bits.", "date": "2014-01-01"},
            {"info": "Le cache L1 du processeur XYZ est de 129 Ko.", "date": "2022-01-01"},
            {"info": "Le cache L1 du processeur XYZ est de 102 Ko.", "date": "2016-01-01"},
            {"info": "Le processeur XYZ possède 8 cœurs physiques et 16 threads.", "date": "2022-01-01"},
            {"info": "Le processeur XYZ possède 4 cœurs physiques et 8 threads.", "date": "2018-01-01"},
            {"info": "Le processeur XYZ est jaune.", "date": "2022-01-01"},
            {"info": "Le processeur XYZ mesure 5 cm.", "date": "2022-01-01"},
            {"info": "Le processeur XYZ mesure 7 cm.", "date": "2018-01-01"},
        ]


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

    start = time.time()

    dataset3=[]
    with open("reglement.txt",'r',encoding="utf-8") as f:
        for ligne in f:
            dataset3.append({"info":ligne.split("\t")[0],"context":ligne.split("\t")[1]})


    knowledge = KnowledgeBase(dataset3,"BAAI/bge-small-en","BAAI/bge-small-en",index_path="faiss_index.idx",load=True,method=1)
    fetcher=VectorFetcher(knowledge)
    nb_queries=5


    def ask(user_query,small_to_big):
        start = time.time()
        print("\n\n---------------------------\n",user_query)
        context=fetcher.retrieve(user_query,num_queries=nb_queries)

        str_context=""
        for i in range(nb_queries):
            if small_to_big:
                str_context+=context[i]["context"]
            else:
                str_context+=context[i]["info"]
        generator=RAGGenerator()
        print(generator.generate(query=user_query,context=str_context))
            
        end = time.time()
        print(f"[ask] Temps d'exécution : {end - start:.2f} secondes")

    user_query="Quel est la capitale de la russie ?"
    ask(user_query,small_to_big=True)

    end = time.time()
    print(f"[global] Temps d'exécution : {end - start:.2f} secondes")
    start = time.time()
    user_query="Quel est le code secret ?"
    ask(user_query,small_to_big=True)

    end = time.time()
    print(f"[global] Temps d'exécution : {end - start:.2f} secondes")