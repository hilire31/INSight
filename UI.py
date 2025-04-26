import streamlit as st
from INSight import RAGDataset, KnowledgeBase, VectorFetcher, UserPrompt
import config
import os

st.set_page_config(page_title="📚 CÉLia - RAG PDF Chatbot", layout="wide")

# Sidebar - Paramètres
st.sidebar.title("Paramètres")
token_model = st.sidebar.text_input("Tokenizer", "BAAI/bge-small-en")
embed_model = st.sidebar.text_input("Embedding Model", "BAAI/bge-small-en")
model = st.sidebar.text_input("Generation Model", config.GENERATOR_MODEL)
num_contextes = st.sidebar.slider("Nombre de contextes", 2, 20, config.NB_CONTEXTES)
chunk_size = st.sidebar.slider("Taille des chunks", 20, 1000, config.CHUNK_MAX_SIZE)

# Titre
st.title("🤖 CÉLia - Chatbot RAG sur vos PDFs")

uploaded_files = st.file_uploader("📄 Uploadez vos fichiers PDF", accept_multiple_files=True, type=["pdf"])

if uploaded_files:
    if st.button("📤 Charger les documents"):
        with st.spinner("Chargement et indexation en cours..."):
            rag_dataset = RAGDataset(load=False, uploaded_files=uploaded_files, chunk_size=chunk_size)
            kb = KnowledgeBase(rag_dataset, token_model, embed_model)
            kb.build_faiss_index()
            vf = VectorFetcher(kb)
            up = UserPrompt(vf)
        st.success("✅ Documents chargés et index construits !")
        st.session_state["up"] = up
        st.session_state["chat_history"] = []


# Interface conversationnelle
if "up" in st.session_state:
    st.divider()
    st.markdown("### 💬 Discussion avec CÉLia")
    
    # Afficher l'historique de la conversation
    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Entrée utilisateur
    user_input = st.chat_input("Pose ta question ici...")
    if user_input:
        # Afficher le message utilisateur
        st.chat_message("user").markdown(user_input)
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        
        # Générer la réponse
        with st.spinner("CÉLia réfléchit..."):
            up = st.session_state["up"]
            response = up.ask(user_input, nb_contextes=num_contextes, model=model)

        # Afficher la réponse de l'assistant
        st.chat_message("assistant").markdown(response)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})
