import faiss
import torch
import numpy as np
from transformers import (AutoTokenizer, AutoModel)


class KnowledgeBase:
    
    def __init__(self,dataset:list,token_embed:AutoTokenizer,model_embed:AutoModel):
        self.setDataset(dataset)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def loadTokeniser(self):
        self.tokenizer_embed = AutoTokenizer.from_pretrained("BAAI/bge-small-en") # tokenize
        self.model_embed = AutoModel.from_pretrained("BAAI/bge-small-en").to(self.device) # vectorize
    def build_faiss_index(self):
        dimension = 384 #vecteur de 384 dimensions pour chaque token
        self.index = faiss.IndexFlatL2(dimension)
        questions = [d["question"] for d in self.dataset]
        embeddings = np.vstack([self.get_embedding(q) for q in questions])
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
    def rewrite(self,query_user):
        pass
    def verify(self,query_user,query):
        pass
    def set_context(self,context):
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
    def __init__(self):
        pass
    def retrieve(self,query:str): 
        pass
    def verify(self,query:str,queries:list):
        pass



class ToolManager:
    class ToolMySQL():
        def retrieve(self,query:str,bdd:str):
            pass
        
    class ToolSem(): 
        pass



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
                self.toolSQL=ToolManager.ToolMySQL()
                bdd="/path/to/database" #???
                self.context.append(self.toolSQL.retrieve(self,query,bdd).context)
        self.generator=RAGGenerator(generator_model='llama3.3')
        self.generator.generate(query=self.query,context=self.context)



myrag=Rag()
user_query="je sasis pas mais je veut savoir qu'elle matière choisirr stpp mais j'aime pas les maths et jeveux 25 ects"
Rag.call(user_query)

