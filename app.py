"""
🎓 TÜ Kursuse Nõustaja – RAG + hübriidotsing + metaandmete filtreerimine + järelvestlus
=========================================================================================
Kasutaja valib metaandmete filtrid Streamlit'i külgribalt.
Pärast hübriidotsingu (semantiline + võtmesõna) tulemusi saab vestlust jätkata.
Rakendus loeb tokeneid ja arvutab jooksvat kulu.
"""

import os
import pickle
import streamlit as st
import pandas as pd
from datetime import datetime
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Lae .env fail, kui see eksisteerib
load_dotenv()

from utils import (
    count_tokens,
    messages_to_token_count,
    log_feedback,
    apply_filters,
    hybrid_search,
    build_system_prompt,
    calculate_cost,
)

# ──────────────────────────────────────────────────────────────
# Konfiguratsioon
# ──────────────────────────────────────────────────────────────
DATA_PATH = "andmed/puhtad_andmed.csv"
EMBEDDINGS_PATH = "andmed/puhtad_andmed_embeddings.pkl"
EMBED_MODEL_NAME = "BAAI/bge-m3"
LLM_MODEL = "google/gemma-3-27b-it"
TOP_K = 5

# Hinnang: OpenRouter google/gemma-3-27b-it
COST_PER_1M_INPUT_TOKENS = 0.10   # USD
COST_PER_1M_OUTPUT_TOKENS = 0.20  # USD

# ──────────────────────────────────────────────────────────────
# Lehe seadistus
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TÜ Kursuse Nõustaja", page_icon="🎓", layout="wide"
)
st.title("🎓 TÜ Kursuse Nõustaja")
st.caption(
    "Hübriidotsing (semantiline + võtmesõna) koos metaandmete filtreerimise ja järelvestlusega"
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
# Külgriba: API-võti · filtrid · otsingu seaded · tokenite info
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    # Lae API võti: .env failist või käsitsi sisestades
    env_key = os.getenv("OPENROUTER_API_KEY", "")
    if env_key:
        api_key = st.text_input(
            "🔑 OpenRouter API Key",
            value=env_key,
            type="password",
            help="Võti laetud .env failist. Saad seda siin üle kirjutada.",
        )
    else:
        api_key = st.text_input(
            "🔑 OpenRouter API Key",
            type="password",
            help=".env faili ei leitud – sisesta võti käsitsi.",
        )

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

    # Õppeaste
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
    st.subheader("⚙️ Otsingu seaded")
    semantic_weight = st.slider(
        "Semantilise otsingu kaal",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Suurem väärtus eelistab tähenduslikku (vektori-) otsingut. "
             "Väiksem väärtus eelistab täpset võtmesõna otsingut.",
    )
    keyword_weight = round(1.0 - semantic_weight, 2)
    st.caption(f"Võtmesõnaotsingu kaal: **{keyword_weight}**")

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

    # ──────────────────────────────────────────────────────────
    # 🎮 Meelelahutuse nurk külgribal
    # ──────────────────────────────────────────────────────────
    st.divider()
    st.subheader("🎮 Meelelahutus")
    st.caption("Mängi, kuni AI vastust ootad!")

    game_choice = st.selectbox(
        "Vali tegevus:",
        ["🐍 Madu mäng", "🎬 Õppevideo"],
        key="game_choice",
    )

    if game_choice == "🐍 Madu mäng":
        st.components.v1.html(
            """
            <style>
                #snake-game-container { text-align: center; font-family: Arial, sans-serif; }
                canvas { border: 2px solid #4a4a4a; border-radius: 8px; background: #1a1a2e; display: block; margin: 0 auto; }
                .score-display { color: #e94560; font-size: 16px; font-weight: bold; margin: 6px 0; }
                .controls { margin-top: 8px; }
                .controls button {
                    width: 40px; height: 40px; font-size: 18px; margin: 2px;
                    border: none; border-radius: 6px; cursor: pointer;
                    background: #16213e; color: #e94560; font-weight: bold;
                }
                .controls button:hover { background: #e94560; color: white; }
                .restart-btn {
                    background: #e94560 !important; color: white !important;
                    width: auto !important; padding: 4px 16px !important;
                    font-size: 13px !important; margin-top: 6px !important;
                }
            </style>
            <div id="snake-game-container">
                <div class="score-display">Skoor: <span id="score">0</span> | Rekord: <span id="highscore">0</span></div>
                <canvas id="snakeCanvas" width="240" height="240"></canvas>
                <div class="controls">
                    <div><button onclick="setDir(0,-1)">▲</button></div>
                    <div>
                        <button onclick="setDir(-1,0)">◄</button>
                        <button onclick="setDir(0,1)">▼</button>
                        <button onclick="setDir(1,0)">►</button>
                    </div>
                    <div><button class="restart-btn" onclick="startGame()">🔄 Uuesti</button></div>
                </div>
            </div>
            <script>
                const cvs = document.getElementById('snakeCanvas');
                const ctx = cvs.getContext('2d');
                const SZ = 15, W = cvs.width/SZ, H = cvs.height/SZ;
                let snake, food, dir, score, hs = 0, alive, loop;
                function startGame() {
                    clearInterval(loop);
                    snake = [{x:8,y:8}]; dir = {x:1,y:0}; score = 0; alive = true;
                    placeFood(); update();
                    loop = setInterval(update, 120);
                    document.getElementById('score').textContent = score;
                }
                function placeFood() {
                    do { food = {x:Math.floor(Math.random()*W), y:Math.floor(Math.random()*H)}; }
                    while(snake.some(s=>s.x===food.x&&s.y===food.y));
                }
                function setDir(x,y) { if(!(dir.x===-x&&dir.y===-y)){dir={x,y};} if(!alive)startGame(); }
                function update() {
                    if(!alive) return;
                    const head = {x:snake[0].x+dir.x, y:snake[0].y+dir.y};
                    if(head.x<0||head.x>=W||head.y<0||head.y>=H||snake.some(s=>s.x===head.x&&s.y===head.y)){
                        alive=false; if(score>hs){hs=score;document.getElementById('highscore').textContent=hs;} return;
                    }
                    snake.unshift(head);
                    if(head.x===food.x&&head.y===food.y){score++;document.getElementById('score').textContent=score;placeFood();}
                    else{snake.pop();}
                    ctx.fillStyle='#1a1a2e';ctx.fillRect(0,0,cvs.width,cvs.height);
                    ctx.fillStyle='#e94560';ctx.beginPath();ctx.arc(food.x*SZ+SZ/2,food.y*SZ+SZ/2,SZ/2-1,0,Math.PI*2);ctx.fill();
                    snake.forEach((s,i)=>{ctx.fillStyle=i===0?'#0f3460':'#16213e';ctx.fillRect(s.x*SZ+1,s.y*SZ+1,SZ-2,SZ-2);});
                }
                document.addEventListener('keydown',e=>{
                    if(e.key==='ArrowUp')setDir(0,-1);if(e.key==='ArrowDown')setDir(0,1);
                    if(e.key==='ArrowLeft')setDir(-1,0);if(e.key==='ArrowRight')setDir(1,0);
                });
                startGame();
            </script>
            """,
            height=380,
        )

    elif game_choice == "🎬 Õppevideo":
        st.caption("Vaata lühikest õppevideot tehisintellektist:")
        st.video("https://www.youtube.com/watch?v=ad79nYk2keg")

# ──────────────────────────────────────────────────────────────
# Vestluse kuvamine koos kapotialuse info ja tagasiside vormidega
# ──────────────────────────────────────────────────────────────
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Debug info ja tagasiside ainult assistendi sõnumitele
        if msg["role"] == "assistant" and "debug_info" in msg:
            debug = msg["debug_info"]

            # 1. Kapoti all (RAG andmed JA süsteemiviip)
            with st.expander("🔍 Vaata kapoti alla (RAG ja filtrid)"):
                st.caption(f"**Aktiivsed filtrid:** {debug.get('filters', 'Info puudub')}")
                st.write(f"Filtrid jätsid andmestikku alles **{debug.get('filtered_count', 0)}** kursust.")

                st.write("**Hübriidotsingu tulemus (Top 5 leitud kursust):**")
                ctx_df = debug.get('context_df')
                if ctx_df is not None and not ctx_df.empty:
                    display_cols = ['unique_ID', 'nimi_et', 'eap', 'semester', 'oppeaste', 'score', 'sem_score', 'kw_score']
                    cols_to_show = [c for c in display_cols if c in ctx_df.columns]
                    st.dataframe(ctx_df[cols_to_show], hide_index=True)
                else:
                    st.warning("Ühtegi kursust ei leitud (kas filtrid olid liiga karmid või andmestik tühi).")

                st.text_area(
                    "LLM-ile saadetud täpne prompt:",
                    debug.get('system_prompt', ''),
                    height=150,
                    disabled=True,
                    key=f"prompt_area_{i}"
                )

            # 2. Tagasiside kogumine
            with st.expander("📝 Hinda vastust (Salvestab logisse)"):
                with st.form(key=f"feedback_form_{i}"):
                    rating = st.radio("Hinnang vastusele:", ["👍 Hea", "👎 Halb"], horizontal=True, key=f"rating_{i}")
                    error_step = st.selectbox(
                        "Kui vastus oli halb, siis mis läks valesti?",
                        ["", "Filtrid olid liiga karmid/valed", "Otsing leidis valed ained (RAG viga)", "LLM hallutsineeris/vastas valesti"],
                        key=f"error_step_{i}"
                    )
                    if st.form_submit_button("Salvesta hinnang"):
                        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        ctx_df = debug.get('context_df')
                        ctx_ids = ctx_df['unique_ID'].tolist() if (ctx_df is not None and not ctx_df.empty and 'unique_ID' in ctx_df.columns) else []
                        ctx_names = ctx_df['nimi_et'].tolist() if (ctx_df is not None and not ctx_df.empty and 'nimi_et' in ctx_df.columns) else []
                        log_feedback(ts, debug.get('user_prompt', ''), debug.get('filters', ''), ctx_ids, ctx_names, msg["content"], rating, error_step)
                        st.success("Tagasiside salvestatud tagasiside_log.csv faili!")

# ──────────────────────────────────────────────────────────────
# Kasutaja sisend → hübriidotsing → LLM vastus → järelvestlus
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
            # Aktiivsete filtrite kirjeldus logimiseks
            current_filters_str = f"EAP:{selected_eap}, Sem:{selected_semester}, Keel:{selected_keel}, Linn:{selected_linn}, Aste:{selected_oppeaste}, Vorm:{selected_veebiope}, Hind:{selected_hindamisviis}"

            # ── 1. Filtreerimine ──────────────────────────────
            with st.spinner("Rakendan filtreid ja otsin kursusi …"):
                filtered_df = apply_filters(
                    df_raw,
                    selected_semester=selected_semester,
                    selected_keel=selected_keel,
                    selected_linn=selected_linn,
                    selected_oppeaste=selected_oppeaste,
                    selected_veebiope=selected_veebiope,
                    selected_hindamisviis=selected_hindamisviis,
                    selected_eap=selected_eap,
                )
                n_filtered = len(filtered_df)

                # ── 2. Hübriidotsing ─────────────────────────
                results, context_text = hybrid_search(
                    query=prompt,
                    df_raw=df_raw,
                    embeddings_matrix=embeddings_matrix,
                    embedder=embedder,
                    filtered_indices=filtered_df.index,
                    top_k=TOP_K,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
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
                        "veebiope", "score", "sem_score", "kw_score",
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
            system_prompt_text = build_system_prompt(
                context_text=context_text,
                n_filtered=n_filtered,
                top_k=TOP_K,
                selected_semester=selected_semester,
                selected_keel=selected_keel,
                selected_linn=selected_linn,
                selected_oppeaste=selected_oppeaste,
                selected_veebiope=selected_veebiope,
                selected_hindamisviis=selected_hindamisviis,
                selected_eap=selected_eap,
                eap_min_val=eap_min_val,
                eap_max_val=eap_max_val,
            )
            system_msg = {
                "role": "system",
                "content": system_prompt_text,
            }
            messages_to_send = [system_msg] + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ]

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
                cost = calculate_cost(input_tokens, output_tokens, COST_PER_1M_INPUT_TOKENS, COST_PER_1M_OUTPUT_TOKENS)

                # Uuenda kumulatiivseid loendureid
                st.session_state.total_input_tokens += input_tokens
                st.session_state.total_output_tokens += output_tokens
                st.session_state.total_cost += cost["total_cost"]

                # Koosta debug info tabel
                results_df_display = results.drop(columns=['embedding'], errors='ignore').copy() if not results.empty else pd.DataFrame()

                # Salvesta vastus ajalukku koos debug infoga
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "debug_info": {
                        "user_prompt": prompt,
                        "filters": current_filters_str,
                        "filtered_count": n_filtered,
                        "context_df": results_df_display,
                        "system_prompt": system_prompt_text,
                    }
                })

                # Kuva (selle vastuse) tokenite info
                st.caption(
                    f"📊 ~{input_tokens:,} sisendtokenit · "
                    f"~{output_tokens:,} väljundtokenit · "
                    f"kulu ~${cost['total_cost']:.6f}  |  "
                    f"Kokku: ~{st.session_state.total_input_tokens:,} in / "
                    f"~{st.session_state.total_output_tokens:,} out / "
                    f"${st.session_state.total_cost:.6f}"
                )
                st.rerun()

            except Exception as e:
                st.error(f"Viga LLM-i päringul: {e}")
