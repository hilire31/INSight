import streamlit as st
from INSight import RAGDataset, KnowledgeBase, VectorFetcher, UserPrompt
import config
import os

st.set_page_config(page_title="RAG PDF QA", layout="wide")

# Sidebar - Paramètres
st.sidebar.title("Paramètres")
token_model = st.sidebar.text_input("Tokenizer", "BAAI/bge-small-en")
embed_model = st.sidebar.text_input("Embedding Model", "BAAI/bge-small-en")
num_contextes = st.sidebar.slider("Nombre de contextes", 2, 10, config.NB_CONTEXTES)

# Titre principal
st.title("📚 Système RAG - Question sur des PDFs")

# Sélection du dossier
data_path = st.text_input("Chemin vers le dossier contenant les PDF")

if data_path and os.path.exists(data_path):
    if st.button("Charger les documents"):
        with st.spinner("Chargement des documents..."):
            rag_dataset = RAGDataset(load=False, data_path=data_path)
            kb = KnowledgeBase(rag_dataset, token_model, embed_model)
            kb.build_faiss_index()
            vf = VectorFetcher(kb)
            up = UserPrompt(vf)
        st.success("Documents chargés et index construit avec succès !")
        st.session_state["up"] = up
else:
    st.warning("Veuillez fournir un chemin valide.")

# Poser une question
if "up" in st.session_state:
    user_query = st.text_area("Pose ta question ici :", height=100)
    if st.button("Poser la question"):
        with st.spinner("Recherche en cours..."):
            up = st.session_state["up"]
            reponse = up.ask(user_query, nb_contextes=num_contextes)
            st.markdown("### 🧾 Réponse générée")
            st.write(reponse)
