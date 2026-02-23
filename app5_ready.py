"""
🎓 AI Kursuse Nõustaja – RAG + metaandmete filtreerimine + järelvestlus
=========================================================================
Variant B: kasutaja valib metaandmete filtrid Streamlit'i külgribalt.
Pärast RAG-otsingu tulemusi saab vestlust jätkata (nt küsida leitud ainete kohta).
Rakendus loeb tokeneid ja arvutab jooksvat kulu.
"""

import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
import tiktoken
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ──────────────────────────────────────────────────────────────
# Konfiguratsioon
# ──────────────────────────────────────────────────────────────
DATA_PATH = "andmed/puhtad_andmed.csv"
EMBEDDINGS_PATH = "andmed/puhtad_andmed_embeddings.pkl"
EMBED_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL = "google/gemma-3-27b-it"
TOP_K = 5

# Hinnang: OpenRouter google/gemma-3-27b-it
# (free tier = $0; tasuline tier ≈ $0.10/1M input, $0.20/1M output).
COST_PER_1M_INPUT_TOKENS = 0.10   # USD
COST_PER_1M_OUTPUT_TOKENS = 0.20  # USD

# tiktoken tokenizer (cl100k_base sobib enamikele mudelitele)
try:
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None

# ──────────────────────────────────────────────────────────────
# Abifunktsioonid
# ──────────────────────────────────────────────────────────────

def count_tokens(text: str) -> int:
    """Loe tokenite arv tiktoken'iga; tagavarana ≈ 1 token / 4 tähemärki."""
    if _ENC is not None:
        return len(_ENC.encode(text))
    return max(1, len(text) // 4)


def messages_to_token_count(messages: list[dict]) -> int:
    """Arvuta sõnumite nimekirja kogu tokenite arv."""
    total = 0
    for m in messages:
        total += 4                                   # role + formatting overhead
        total += count_tokens(m.get("content", ""))
    total += 2                                       # lõpetav token
    return total


# ──────────────────────────────────────────────────────────────
# Lehe seadistus
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Kursuse Nõustaja", page_icon="🎓", layout="wide"
)
st.title("🎓 AI Kursuse Nõustaja")
st.caption(
    "RAG süsteem koos metaandmete filtreerimise ja järelvestlusega  ·  Variant B"
)

# ──────────────────────────────────────────────────────────────
# Andmete ja mudelite laadimine (cache)
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Laen embedding-mudelit …")
def load_embedder():
    return SentenceTransformer(EMBED_MODEL_NAME)


@st.cache_resource(show_spinner="Laen andmestikku ja vektoreid …")
def load_data_and_embeddings():
    df = pd.read_csv(DATA_PATH)

    if os.path.exists(EMBEDDINGS_PATH):
        with open(EMBEDDINGS_PATH, "rb") as f:
            embeddings = pickle.load(f)
    else:
        # Esimene käivitus – arvutab vektorid (~1-2 min)
        _embedder = load_embedder()
        texts = []
        for _, row in df.iterrows():
            parts = []
            for col in ["nimi_et", "nimi_en", "kirjeldus", "eesmargid", "opivaljundid"]:
                val = row.get(col)
                if pd.notna(val):
                    parts.append(str(val))
            texts.append(" | ".join(parts))

        embeddings = _embedder.encode(
            texts, show_progress_bar=True, batch_size=32
        )
        with open(EMBEDDINGS_PATH, "wb") as f:
            pickle.dump(embeddings, f)

    return df, embeddings


embedder = load_embedder()
df_raw, embeddings_matrix = load_data_and_embeddings()

# ──────────────────────────────────────────────────────────────
# Sessiooni muutujad
# ──────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "last_results_df" not in st.session_state:
    st.session_state.last_results_df = None

# ──────────────────────────────────────────────────────────────
# Külgriba: API-võti · filtrid · tokenite info
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    api_key = st.text_input("🔑 OpenRouter API Key", type="password")

    st.divider()
    st.subheader("📋 Metaandmete filtrid")
    st.caption("Jäta tühjaks, kui filter pole oluline.")

    # Semester
    semester_options = sorted(df_raw["semester"].dropna().unique().tolist())
    selected_semester = st.multiselect("Semester", semester_options)

    # Õppekeel
    keel_options = sorted(df_raw["keel"].dropna().unique().tolist())
    selected_keel = st.multiselect("Õppekeel", keel_options)

    # Linn
    linn_options = sorted(df_raw["linn"].dropna().unique().tolist())
    selected_linn = st.multiselect("Linn", linn_options)

    # Õppeaste (välja väärtus võib olla mitme komaga eraldatud)
    oppeaste_flat: set[str] = set()
    for vals in df_raw["oppeaste"].dropna().unique():
        for v in str(vals).split(","):
            oppeaste_flat.add(v.strip())
    oppeaste_options = sorted(oppeaste_flat)
    selected_oppeaste = st.multiselect("Õppeaste", oppeaste_options)

    # Õppevorm
    veebiope_options = sorted(df_raw["veebiope"].dropna().unique().tolist())
    selected_veebiope = st.multiselect("Õppevorm", veebiope_options)

    # EAP vahemik
    eap_min_val = float(df_raw["eap"].min())
    eap_max_val = float(df_raw["eap"].max())
    selected_eap = st.slider(
        "EAP vahemik",
        min_value=eap_min_val,
        max_value=eap_max_val,
        value=(eap_min_val, eap_max_val),
        step=0.5,
    )

    # Hindamisviis
    hindamisviis_options = sorted(
        df_raw["hindamisviis"].dropna().unique().tolist()
    )
    selected_hindamisviis = st.multiselect("Hindamisviis", hindamisviis_options)

    st.divider()
    st.subheader("💰 Tokenid ja kulu")
    col1, col2 = st.columns(2)
    col1.metric("Sisend", f"{st.session_state.total_input_tokens:,}")
    col2.metric("Väljund", f"{st.session_state.total_output_tokens:,}")
    st.metric("Jooksev kulu (USD)", f"${st.session_state.total_cost:.6f}")

    st.divider()
    if st.button("🔄 Uus vestlus"):
        st.session_state.messages = []
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.total_cost = 0.0
        st.session_state.last_results_df = None
        st.rerun()


# ──────────────────────────────────────────────────────────────
# Filtreerimisloogika
# ──────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    """Rakenda külgriba filtrid andmetabelile."""
    mask = pd.Series(True, index=df.index)

    if selected_semester:
        mask &= df["semester"].isin(selected_semester)
    if selected_keel:
        mask &= df["keel"].isin(selected_keel)
    if selected_linn:
        mask &= df["linn"].isin(selected_linn)
    if selected_oppeaste:
        mask &= df["oppeaste"].apply(
            lambda x: any(opt in str(x) for opt in selected_oppeaste)
            if pd.notna(x)
            else False
        )
    if selected_veebiope:
        mask &= df["veebiope"].isin(selected_veebiope)
    if selected_hindamisviis:
        mask &= df["hindamisviis"].isin(selected_hindamisviis)
    mask &= (df["eap"] >= selected_eap[0]) & (df["eap"] <= selected_eap[1])

    return df[mask]


# ──────────────────────────────────────────────────────────────
# Semantiline otsing
# ──────────────────────────────────────────────────────────────

def semantic_search(query: str, filtered_indices: pd.Index, top_k: int = TOP_K):
    """Vektorotsing filtreeritud alamhulgal."""
    if len(filtered_indices) == 0:
        return pd.DataFrame(), "Ühtegi kursust ei vasta filtritele."

    idx_positions = [df_raw.index.get_loc(i) for i in filtered_indices]
    filtered_embeddings = embeddings_matrix[idx_positions]

    query_vec = embedder.encode([query])[0]
    scores = cosine_similarity([query_vec], filtered_embeddings)[0]

    filtered_df = df_raw.loc[filtered_indices].copy()
    filtered_df["score"] = scores
    results = filtered_df.sort_values("score", ascending=False).head(top_k)

    # Koosta konteksttekst LLM-ile
    display_cols = [c for c in results.columns if c != "score"]
    context_rows = []
    for _, row in results.iterrows():
        parts = []
        for col in display_cols:
            val = row[col]
            if pd.notna(val):
                parts.append(f"{col}: {val}")
        context_rows.append("\n".join(parts))
    context_text = "\n\n---\n\n".join(context_rows)

    return results, context_text


# ──────────────────────────────────────────────────────────────
# Süsteemiprompt
# ──────────────────────────────────────────────────────────────

def build_system_prompt(context_text: str, n_filtered: int) -> str:
    """Koosta süsteemiprompt aktiivsetest filtritest ja otsingutulemustega."""
    active_filters = []
    if selected_semester:
        active_filters.append(f"Semester: {', '.join(selected_semester)}")
    if selected_keel:
        active_filters.append(f"Õppekeel: {', '.join(selected_keel)}")
    if selected_linn:
        active_filters.append(f"Linn: {', '.join(selected_linn)}")
    if selected_oppeaste:
        active_filters.append(f"Õppeaste: {', '.join(selected_oppeaste)}")
    if selected_veebiope:
        active_filters.append(f"Õppevorm: {', '.join(selected_veebiope)}")
    if selected_hindamisviis:
        active_filters.append(f"Hindamisviis: {', '.join(selected_hindamisviis)}")
    if selected_eap != (eap_min_val, eap_max_val):
        active_filters.append(f"EAP: {selected_eap[0]}–{selected_eap[1]}")

    filter_info = (
        "\n".join(active_filters) if active_filters else "Filtreid pole rakendatud."
    )

    return f"""Oled Tartu Ülikooli ainekursuste nõustaja-vestlusbot.

AKTIIVSED FILTRID (kasutaja valinud):
{filter_info}

Filtritele vastas kokku {n_filtered} kursust. Semantilise otsingu alusel parimad {TOP_K} vastet:

{context_text}

JUHISED:
- Vasta ALATI eesti keeles (v.a juhul, kui kasutaja kirjutab inglise keeles).
- Kasuta ülaltoodud otsingutulemusi oma vastuse alusena.
- Kui otsingutulemused on tühjad, ütle, et pole sobivaid aineid leitud, ja soovita filtreid laiendada.
- Kirjelda soovitatud aineid lühidalt ja selgita, miks need kasutaja küsimusele vastavad.
- Kui kasutaja küsib lisainfot konkreetse leitud aine kohta, anna seda tulemuste põhjal.
- Ära leiuta infot, mida otsingutulemused ei sisalda.
- Kui kasutaja küsib midagi, mis ei puutu kursustesse, ütle viisakalt, et saad aidata ainult kursuste leidmisel."""


# ──────────────────────────────────────────────────────────────
# Vestluse kuvamine
# ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ──────────────────────────────────────────────────────────────
# Kasutaja sisend → RAG → LLM vastus → järelvestlus
# ──────────────────────────────────────────────────────────────

if prompt := st.chat_input("Kirjelda, mida soovid õppida …"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not api_key:
            err = "⚠️ Palun sisesta OpenRouter API võti külgribal!"
            st.error(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
        else:
            # ── 1. Filtreerimine ──────────────────────────────
            with st.spinner("Rakendan filtreid ja otsin kursusi …"):
                filtered_df = apply_filters(df_raw)
                n_filtered = len(filtered_df)

                # ── 2. Semantiline otsing ─────────────────────
                results, context_text = semantic_search(
                    prompt, filtered_df.index
                )
                st.session_state.last_results_df = results

            # Kuva otsingutulemused laiendatavas paneelis
            if not results.empty:
                with st.expander(
                    f"🔍 {len(results)} parimat vastet "
                    f"({n_filtered} kursust vastas filtritele)",
                    expanded=False,
                ):
                    show_cols = [
                        "aine_kood", "nimi_et", "nimi_en", "eap",
                        "semester", "keel", "linn", "oppeaste",
                        "veebiope", "score",
                    ]
                    show_cols = [c for c in show_cols if c in results.columns]
                    st.dataframe(
                        results[show_cols].reset_index(drop=True),
                        use_container_width=True,
                    )
            else:
                st.warning(
                    f"Filtritele vastas {n_filtered} kursust – "
                    "sobivaid tulemusi ei leitud. Proovi filtreid laiendada."
                )

            # ── 3. LLM vastus ────────────────────────────────
            system_msg = {
                "role": "system",
                "content": build_system_prompt(context_text, n_filtered),
            }
            messages_to_send = [system_msg] + st.session_state.messages

            # Sisendtokenite arv
            input_tokens = messages_to_token_count(messages_to_send)

            try:
                client = OpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=api_key,
                )
                stream = client.chat.completions.create(
                    model=LLM_MODEL,
                    messages=messages_to_send,
                    stream=True,
                )
                response_text = st.write_stream(stream)

                # Väljundtokenite arv
                output_tokens = count_tokens(response_text)

                # Kulu
                input_cost = (
                    input_tokens * COST_PER_1M_INPUT_TOKENS / 1_000_000
                )
                output_cost = (
                    output_tokens * COST_PER_1M_OUTPUT_TOKENS / 1_000_000
                )
                msg_cost = input_cost + output_cost

                # Uuenda kumulatiivseid loendureid
                st.session_state.total_input_tokens += input_tokens
                st.session_state.total_output_tokens += output_tokens
                st.session_state.total_cost += msg_cost

                # Salvesta vastus ajalukku
                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

                # Kuva (selle vastuse) tokenite info
                st.caption(
                    f"📊 ~{input_tokens:,} sisendtokenit · "
                    f"~{output_tokens:,} väljundtokenit · "
                    f"kulu ~${msg_cost:.6f}  |  "
                    f"Kokku: ~{st.session_state.total_input_tokens:,} in / "
                    f"~{st.session_state.total_output_tokens:,} out / "
                    f"${st.session_state.total_cost:.6f}"
                )

            except Exception as e:
                st.error(f"Viga LLM-i päringul: {e}")