"""
Abifunktsioonid AI Kursuse Nõustaja rakendusele.
Eraldatud testide ja korduvkasutuse jaoks.
"""

import os
import csv
import re
import numpy as np
import pandas as pd
import tiktoken
from sklearn.metrics.pairwise import cosine_similarity

# tiktoken tokenizer (cl100k_base sobib enamikele mudelitele)
try:
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENC = None


# ──────────────────────────────────────────────────────────────
# Tokenite loendamine
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
# Tagasiside logimise funktsioon
# ──────────────────────────────────────────────────────────────

def log_feedback(timestamp, prompt, filters, context_ids, context_names, response, rating, error_category, file_path='tagasiside_log.csv'):
    """Salvesta tagasiside CSV-faili."""
    file_exists = os.path.isfile(file_path)

    with open(file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Aeg', 'Kasutaja päring', 'Filtrid', 'Leitud ID-d', 'Leitud ained', 'LLM Vastus', 'Hinnang', 'Veatüüp'])
        writer.writerow([timestamp, prompt, filters, str(context_ids), str(context_names), response, rating, error_category])


# ──────────────────────────────────────────────────────────────
# Filtreerimisloogika
# ──────────────────────────────────────────────────────────────

def apply_filters(
    df: pd.DataFrame,
    selected_semester: list = None,
    selected_keel: list = None,
    selected_linn: list = None,
    selected_oppeaste: list = None,
    selected_veebiope: list = None,
    selected_hindamisviis: list = None,
    selected_eap: tuple = None,
) -> pd.DataFrame:
    """Rakenda metaandmete filtrid andmetabelile.
    
    Kõik filtrid on valikulised – tühi nimekiri või None tähendab, 
    et filtrit ei rakendata.
    """
    selected_semester = selected_semester or []
    selected_keel = selected_keel or []
    selected_linn = selected_linn or []
    selected_oppeaste = selected_oppeaste or []
    selected_veebiope = selected_veebiope or []
    selected_hindamisviis = selected_hindamisviis or []

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
    if selected_eap is not None:
        mask &= (df["eap"] >= selected_eap[0]) & (df["eap"] <= selected_eap[1])

    return df[mask]


# ──────────────────────────────────────────────────────────────
# Võtmesõna otsing (BM25-stiilis)
# ──────────────────────────────────────────────────────────────

def keyword_search(query: str, df: pd.DataFrame, text_column: str = "kirjeldus", top_k: int = 5) -> pd.DataFrame:
    """Lihtne TF-põhine võtmesõnaotsing toetamaks hübriidotsingut.
    
    Otsib kasutaja päringu sõnu veergudest: nimi_et, nimi_en, kirjeldus, 
    eesmargid, opivaljundid. Tagastab top_k tulemust skoori järgi.
    """
    if df.empty:
        return df.copy()

    search_cols = ["nimi_et", "nimi_en", "kirjeldus", "eesmargid", "opivaljundid"]
    search_cols = [c for c in search_cols if c in df.columns]

    # Puhasta päring ja tee sõnad
    query_lower = query.lower().strip()
    query_words = [w for w in re.split(r'\s+', query_lower) if len(w) >= 2]

    if not query_words:
        return df.head(0)

    scores = np.zeros(len(df))

    for col in search_cols:
        col_values = df[col].fillna("").astype(str).str.lower()
        for word in query_words:
            # Iga sõna tabamus annab 1 punkti iga sõna esinemise eest veerus
            matches = col_values.str.count(re.escape(word))
            scores += matches.values

    result = df.copy()
    result["keyword_score"] = scores

    # Filtreeri välja nullskooriga read
    result = result[result["keyword_score"] > 0]
    result = result.sort_values("keyword_score", ascending=False).head(top_k)

    return result


# ──────────────────────────────────────────────────────────────
# Semantiline otsing
# ──────────────────────────────────────────────────────────────

def semantic_search_fn(
    query: str,
    df_raw: pd.DataFrame,
    embeddings_matrix: np.ndarray,
    embedder,
    filtered_indices: pd.Index,
    top_k: int = 5,
):
    """Vektorotsing filtreeritud alamhulgal.
    
    Tagastab: (results_df, context_text)
    """
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
# Hübriidotsing: semantiline + võtmesõna
# ──────────────────────────────────────────────────────────────

def hybrid_search(
    query: str,
    df_raw: pd.DataFrame,
    embeddings_matrix: np.ndarray,
    embedder,
    filtered_indices: pd.Index,
    top_k: int = 5,
    semantic_weight: float = 0.7,
    keyword_weight: float = 0.3,
):
    """Hübriidotsing: kombineerib semantilist ja võtmesõna otsingut.
    
    semantic_weight + keyword_weight peaksid andma 1.0.
    Tagastab: (results_df, context_text)
    """
    if len(filtered_indices) == 0:
        return pd.DataFrame(), "Ühtegi kursust ei vasta filtritele."

    filtered_df = df_raw.loc[filtered_indices].copy()

    # 1. Semantiline otsing – arvuta skoorid kõigile filtreeritud ridadele
    idx_positions = [df_raw.index.get_loc(i) for i in filtered_indices]
    filtered_embeddings = embeddings_matrix[idx_positions]
    query_vec = embedder.encode([query])[0]
    sem_scores = cosine_similarity([query_vec], filtered_embeddings)[0]

    # Normaliseeri semantilised skoorid [0, 1]
    sem_min, sem_max = sem_scores.min(), sem_scores.max()
    if sem_max > sem_min:
        sem_scores_norm = (sem_scores - sem_min) / (sem_max - sem_min)
    else:
        sem_scores_norm = np.zeros_like(sem_scores)

    # 2. Võtmesõnaotsing
    kw_result = keyword_search(query, filtered_df, top_k=len(filtered_df))
    kw_scores = np.zeros(len(filtered_df))
    if not kw_result.empty and "keyword_score" in kw_result.columns:
        kw_max = kw_result["keyword_score"].max()
        if kw_max > 0:
            for i, idx in enumerate(filtered_df.index):
                if idx in kw_result.index:
                    kw_scores[i] = kw_result.loc[idx, "keyword_score"] / kw_max

    # 3. Kombineeri
    combined_scores = semantic_weight * sem_scores_norm + keyword_weight * kw_scores

    filtered_df["score"] = combined_scores
    filtered_df["sem_score"] = sem_scores
    filtered_df["kw_score"] = kw_scores
    results = filtered_df.sort_values("score", ascending=False).head(top_k)

    # Koosta konteksttekst LLM-ile
    display_cols = [c for c in results.columns if c not in ("score", "sem_score", "kw_score")]
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

def build_system_prompt(
    context_text: str,
    n_filtered: int,
    top_k: int = 5,
    selected_semester: list = None,
    selected_keel: list = None,
    selected_linn: list = None,
    selected_oppeaste: list = None,
    selected_veebiope: list = None,
    selected_hindamisviis: list = None,
    selected_eap: tuple = None,
    eap_min_val: float = 0.0,
    eap_max_val: float = 60.0,
) -> str:
    """Koosta süsteemiprompt aktiivsetest filtritest ja otsingutulemustega."""
    selected_semester = selected_semester or []
    selected_keel = selected_keel or []
    selected_linn = selected_linn or []
    selected_oppeaste = selected_oppeaste or []
    selected_veebiope = selected_veebiope or []
    selected_hindamisviis = selected_hindamisviis or []

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
    if selected_eap is not None and selected_eap != (eap_min_val, eap_max_val):
        active_filters.append(f"EAP: {selected_eap[0]}–{selected_eap[1]}")

    filter_info = (
        "\n".join(active_filters) if active_filters else "Filtreid pole rakendatud."
    )

    return f"""Oled Tartu Ülikooli ainekursuste nõustaja-vestlusbot. Sinu nimi on „TÜ Kursuse Nõustaja".

AKTIIVSED FILTRID (kasutaja valinud):
{filter_info}

Filtritele vastas kokku {n_filtered} kursust. Semantilise ja võtmesõnaotsingu alusel parimad {top_k} vastet:

{context_text}

JUHISED:
- Vasta ALATI eesti keeles (v.a juhul, kui kasutaja kirjutab inglise keeles).
- Kasuta ülaltoodud otsingutulemusi oma vastuse alusena. Tulemused on järjestatud asjakohasuse järgi.
- Kui otsingutulemused on tühjad, ütle ausalt, et pole sobivaid aineid leitud, ja soovita filtreid laiendada.
- Nimeta alati aine kood (nt MTAT.03.263), täisnimi ja EAP maht.
- Kirjelda soovitatud aineid lühidalt ja selgita, miks need kasutaja küsimusele vastavad.
- Kui kasutaja küsib lisainfot konkreetse leitud aine kohta, anna seda tulemuste põhjal.
- Ära leiuta ega hallutsinerimine: kui andmeid pole, ütle seda ausalt.
- Ära tekita aineid, mida otsingutulemused ei sisalda.
- Kui kasutaja küsib midagi, mis ei puutu kursustesse, ütle viisakalt, et saad aidata ainult kursuste leidmisel.
- Vorminda vastus selgelt ja struktureeritult (kasuta nummerdatud loendit).
- Maini ka eeldusaineid, kui need on olemas."""


# ──────────────────────────────────────────────────────────────
# Kulu arvutamine
# ──────────────────────────────────────────────────────────────

def calculate_cost(input_tokens: int, output_tokens: int, cost_per_1m_input: float = 0.10, cost_per_1m_output: float = 0.20) -> dict:
    """Arvuta LLM päringu kulu.
    
    Tagastab sõnastiku: input_cost, output_cost, total_cost
    """
    input_cost = input_tokens * cost_per_1m_input / 1_000_000
    output_cost = output_tokens * cost_per_1m_output / 1_000_000
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }
