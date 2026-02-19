import streamlit as st
from openai import OpenAI

# Iluasjad: pealkiri, allkiri
st.title("🎓 AI Kursuse Nõustaja - Samm 2")
st.caption("Vestlus päris tehisintellektiga (Gemma 3).")

# UUS 
# Külgriba API võtme jaoks (sidebar)
with st.sidebar:
    api_key = st.text_input("OpenRouter API võti", type="password", placeholder="sk-or-...")

# 1. Algatame vestluse ajaloo, kui seda veel pole
if "messages" not in st.session_state:
    st.session_state.messages = []

# 2. Kuvame vestluse senise ajaloo (History)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 3. Korjame üles uue kasutaja sisendi
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    #kuvame kohe kasutaja sõnumi
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # defineerime süsteemiprompti, 
    # genereerime vastuse, kontrollime ka, et kas võti on olemas (külgriba väljund)
    # kuvame vastuse striimina, ilmub jooksvalt
    system_prompt = "Sa oled abivalmis kursuse nõustaja, kes aitab tudengitel valida aineid ja koostada õppekavasid. Vasta alati eesti keeles."

    with st.chat_message("assistant"):
        if not api_key:
            error_msg = "⚠️ Palun sisesta OpenRouter API võti külgribale."
            st.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            #defineeri OpenAI klient, anna sellele süsteemiprompt ja vestluse ajalugu
            client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )
                        
            try:
                # Kasutame OpenAI kliendi võimalust striimimida, et vastus ilmuks jooksvalt
                stream = client.chat.completions.create(
                    model="google/gemma-3-27b-it",
                    messages=[
                        {"role": "user", "content": f"[Juhis]: {system_prompt}"},
                        *st.session_state.messages,
                    ],
                    stream=True,
                )
                
                response = st.write_stream(
                    (chunk.choices[0].delta.content or "" for chunk in stream if chunk.choices[0].delta.content is not None)
                )
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Viga: {e}")