import streamlit as st
import pandas as pd
import numpy as np
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# pealkiri
st.title("🎓 AI Kursuse Nõustaja - Samm 5")
st.caption("RAG süsteem koos eel-filtreerimisega.")

# külgriba
with st.sidebar:
    api_key = st.text_input("OpenRouter API Key", type="password")
    st.info("Selles versioonis on koodis filter: ainult ingliskeelsed kursused.")

# embed mudel, täisandmestik ja vektorandmebaas läheb cache'i
@st.cache_resource
def get_models():
    embedder = SentenceTransformer("BAAI/bge-m3")
    df = pd.read_csv("../data/puhtad_andmed.csv")
    embeddings_df = pd.read_pickle("../data/puhtad_andmed_embeddings.pkl")
    return embedder, df, embeddings_df
embedder, df, embeddings_df = get_models()

# 1. alustame
if "messages" not in st.session_state:
    st.session_state.messages = []
# 2. kuvame ajaloo
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. kuulame kasutaja sõnumit
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "Palun sisesta API võti!"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            # UUS ÜLESANNE: Filtreerimine enne semantilist otsingut
            with st.spinner("Otsin sobivaid kursusi..."):
                # 1. ühenda kaks andmetabelit ja filtreeri esmalt EAP-de ja semestri alusel
                merged_df = pd.merge(df, embeddings_df, on='unique_ID')
                mask = 
                filtered_df = 

                #kontroll (sanity check)
                if filtered_df.empty:
                    st.warning("Ühtegi kursust ei vasta filtritele.")
                    context_text = "Sobivaid kursusi ei leitud."
                else:
                    # Arvutame sarnasuse ja sorteerime tabeli
                    query_vec = embedder.encode([prompt])[0]
                    # lisa embedding andmefreimile score rida
                    filtered_df['score'] = cosine_similarity([query_vec], filtered_df['embedding'])[0]
                    
                    # Leiame 5 kõige sarnasemat (suurim skoor)
                    return_N = 5
                    results_df = filtered_df.sort_values('score', ascending=False).head(return_N)
                    results_df.drop(['score', 'embedding'], axis=1, inplace=True)
                    context_text = results_df.to_string()

                # 3. LLM vastus
                client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
                system_prompt = {
                    "role": "system", 
                    "content": f"Oled nõustaja. Kasuta järgmisi kursusi (filtreeritud: inglise keel):\n\n{context_text}"
                }
                
                messages_to_send = [system_prompt] + st.session_state.messages
                
                try:
                    stream = client.chat.completions.create(
                        model="google/gemma-3-27b-it:free",
                        messages=messages_to_send,
                        stream=True
                    )
                    response = st.write_stream(stream)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Viga: {e}")