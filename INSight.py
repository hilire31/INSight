import faiss
import torch
import numpy as np
from transformers import (AutoTokenizer, AutoModel)
from langchain_ollama import OllamaLLM

class KnowledgeBase:
    
    def __init__(self,dataset:list,token_embed:AutoTokenizer,model_embed:AutoModel):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loadTokeniser(token_embed,model_embed)
        self.setDataset(dataset)
        
        self.build_faiss_index()

    def loadTokeniser(self,token_embed:AutoTokenizer,model_embed:AutoModel):
        self.tokenizer_embed = AutoTokenizer.from_pretrained(token_embed) # tokenize
        self.model_embed = AutoModel.from_pretrained(model_embed).to(self.device) # vectorize
    def build_faiss_index(self):
        dimension = 384 #vecteur de 384 dimensions pour chaque token
        self.index = faiss.IndexFlatL2(dimension)
        embeddings = np.vstack([self.get_embedding(q["info"]) for q in self.dataset])
        self.index.add(embeddings)
        faiss.write_index(self.index, "faiss_index.idx")
        
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
    def retrieve(self,query:str): 
        query_embedding = self.knowledge.get_embedding(query)
        D, I = self.knowledge.index.search(query_embedding, k=1)
        retrieved_info = self.knowledge.dataset[I[0][0]]
        input_text = f"context: {retrieved_info["info"]} question: {query}"
        print("input : ",input_text)
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
            self.chain = GraphSparqlQAChain.from_llm(llm, graph=self.graph, verbose=True,allow_dangerous_requests=True, return_sparql_query= True)

        def ask(self,question:str):
            response = self.chain(question)
            return response



class RAGGenerator:
    def __init__(self,generator_model:str):
        pass
    def generate(self,query:str,context:str):
        pass

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


"""
myrag=Rag()
user_query="je sasis pas mais je veut savoir qu'elle matière choisirr stpp mais j'aime pas les maths et jeveux 25 ects"
Rag.call(user_query)
"""
dataset = [
            {"info": "Les registres du processeur XYZ ont une taille de 68 bits."},
            {"info": "Le cache L1 du processeur XYZ est de 129 Ko."},
            {"info": "Le processeur XYZ possède 8 cœurs physiques et 16 threads."},
            {"info": "Le processeur XYZ est jaune."},
            {"info": "Le processeur XYZ mesure 5 cm."},
        ]

knowledge = KnowledgeBase(dataset,"BAAI/bge-small-en","BAAI/bge-small-en")
fetcher=VectorFetcher(knowledge)
fetcher.retrieve("taille du cache")