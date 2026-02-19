import streamlit as st

# Iluasjad: pealkiri, alapealkiri
st.title("🎓 AI Kursuse Nõustaja")
st.caption("Lihtne vestlusliides automaatvastusega.")

# 1. Algatame vestluse ajaloo, kui seda veel pole


# 2. Kuvame vestluse senise ajaloo (History)


# 3. Korjame üles uue kasutaja sisendi (Action)
if prompt := st.chat_input("Kirjelda, mida soovid õppida..."):
    # Kuvame kohe kasutaja sõnumi ja salvestame selle ka ajalukku

    # Kuvame vastuse ja salvestame ajalukku
    response = "LLM pole veel ühendatud, see on automaatvastus."