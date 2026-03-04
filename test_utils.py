"""
Testid AI Kursuse Nõustaja rakenduse abifunktsioonidele.
Käivitamine: pytest test_utils.py -v
"""

import os
import csv
import tempfile
import numpy as np
import pandas as pd
import pytest

from utils import (
    count_tokens,
    messages_to_token_count,
    log_feedback,
    apply_filters,
    keyword_search,
    semantic_search_fn,
    hybrid_search,
    build_system_prompt,
    calculate_cost,
)


# ══════════════════════════════════════════════════════════════
# Fixture: näidis-DataFrame (kursuste andmed)
# ══════════════════════════════════════════════════════════════

@pytest.fixture
def sample_df():
    """Loo testide jaoks väike näidisandmestik."""
    data = {
        "unique_ID": ["MTAT.03.263", "LTAT.01.001", "LTAT.02.002", "FLEE.06.100", "MJRI.09.027"],
        "aine_kood": ["MTAT.03.263", "LTAT.01.001", "LTAT.02.002", "FLEE.06.100", "MJRI.09.027"],
        "nimi_et": [
            "Tehisintellekt I",
            "Andmeteaduse alused",
            "Masinõpe",
            "Saksa keel A1",
            "Sissejuhatus majandusteooriasse",
        ],
        "nimi_en": [
            "Artificial Intelligence I",
            "Fundamentals of Data Science",
            "Machine Learning",
            "German A1",
            "Introduction to Economics",
        ],
        "eap": [6.0, 6.0, 6.0, 3.0, 6.0],
        "semester": ["sügis", "kevad", "sügis", "kevad", "kevad"],
        "hindamisviis": [
            "Eristav (A, B, C, D, E, F, mi)",
            "Eristav (A, B, C, D, E, F, mi)",
            "Eristav (A, B, C, D, E, F, mi)",
            "Eristamata (arv, m.arv, mi)",
            "Eristav (A, B, C, D, E, F, mi)",
        ],
        "keel": ["eesti keel", "eesti keel", "inglise keel", "eesti keel", "eesti keel"],
        "linn": ["Tartu linn", "Tartu linn", "Tartu linn", "Tartu linn", "Tartu linn"],
        "oppeaste": [
            "bakalaureuseõpe, magistriõpe",
            "bakalaureuseõpe",
            "magistriõpe",
            "bakalaureuseõpe",
            "bakalaureuseõpe",
        ],
        "veebiope": ["põimõpe", "põimõpe", "veebiõpe", "põimõpe", "põimõpe"],
        "kirjeldus": [
            "Kursus käsitleb tehisintellekti põhialuseid, otsingualgoritme ja masinõpet.",
            "Andmeteaduse alused katab statistika, andmeanalüüsi ja visualiseerimist.",
            "Masinõppe kursus käsitleb juhendatud ja juhendamata õpet, närvivõrke.",
            "Saksa keele algkursus, grammatika ja sõnavara.",
            "Sissejuhatus mikro- ja makroökonoomikasse, turuteooria ja riigi roll.",
        ],
        "eesmargid": [
            "Tutvustada tehisintellekti meetodeid ja algoritme.",
            "Õpetada andmeteaduse põhimõtteid ja tööriistu.",
            "Omandada masinõppe tehnikad.",
            "Õppida saksa keele algteadmised.",
            "Tutvustada majandusteooria aluseid.",
        ],
        "opivaljundid": [
            "Üliõpilane oskab rakendada AI algoritme.",
            "Üliõpilane oskab analüüsida andmeid Pythoniga.",
            "Üliõpilane suudab treenida masinõppe mudeleid.",
            "Üliõpilane oskab suhelda saksa keeles algtasemel.",
            "Üliõpilane mõistab majanduse toimimise mehhanisme.",
        ],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_embeddings(sample_df):
    """Loo juhuslikud embedding'id näidisandmestiku jaoks."""
    np.random.seed(42)
    return np.random.rand(len(sample_df), 64).astype(np.float32)


class FakeEmbedder:
    """Lihtne mock-embedder testide jaoks."""
    def encode(self, texts):
        np.random.seed(123)
        if isinstance(texts, str):
            texts = [texts]
        return np.random.rand(len(texts), 64).astype(np.float32)


# ══════════════════════════════════════════════════════════════
# Testid: count_tokens
# ══════════════════════════════════════════════════════════════

class TestCountTokens:
    def test_empty_string(self):
        result = count_tokens("")
        assert result >= 0

    def test_short_string(self):
        result = count_tokens("Tere")
        assert result >= 1

    def test_longer_string(self):
        result = count_tokens("See on pikem tekst, mis sisaldab mitut sõna ja lauset.")
        assert result > 1

    def test_returns_integer(self):
        result = count_tokens("test")
        assert isinstance(result, int)


# ══════════════════════════════════════════════════════════════
# Testid: messages_to_token_count
# ══════════════════════════════════════════════════════════════

class TestMessagesToTokenCount:
    def test_empty_list(self):
        result = messages_to_token_count([])
        assert result == 2  # ainult lõpetav token

    def test_single_message(self):
        msgs = [{"role": "user", "content": "Tere"}]
        result = messages_to_token_count(msgs)
        assert result > 2  # vähemalt overhead + sisutokenid + lõpp

    def test_multiple_messages(self):
        msgs = [
            {"role": "system", "content": "Oled abiline assistent."},
            {"role": "user", "content": "Mis on masinõpe?"},
        ]
        result = messages_to_token_count(msgs)
        single_result = messages_to_token_count([msgs[0]])
        assert result > single_result

    def test_message_without_content(self):
        msgs = [{"role": "user"}]
        result = messages_to_token_count(msgs)
        assert result >= 6  # 4 (overhead) + 0 (content) + 2 (lõpp)


# ══════════════════════════════════════════════════════════════
# Testid: log_feedback
# ══════════════════════════════════════════════════════════════

class TestLogFeedback:
    def test_creates_file_with_header(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            tmp_path = f.name
        # Kustuta fail, et testida loomist
        os.remove(tmp_path)

        try:
            log_feedback(
                "2026-01-01 10:00:00",
                "Tere",
                "EAP:(0,60)",
                ["ID1"],
                ["Aine1"],
                "Vastus",
                "👍 Hea",
                "",
                file_path=tmp_path,
            )
            assert os.path.exists(tmp_path)

            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 2  # päis + 1 rida
            assert rows[0][0] == 'Aeg'
            assert rows[1][0] == '2026-01-01 10:00:00'
        finally:
            os.remove(tmp_path)

    def test_appends_to_existing_file(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            tmp_path = f.name
        os.remove(tmp_path)

        try:
            log_feedback("2026-01-01", "Q1", "", [], [], "A1", "👍 Hea", "", file_path=tmp_path)
            log_feedback("2026-01-02", "Q2", "", [], [], "A2", "👎 Halb", "RAG viga", file_path=tmp_path)

            with open(tmp_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                rows = list(reader)

            assert len(rows) == 3  # päis + 2 rida
        finally:
            os.remove(tmp_path)

    def test_handles_unicode(self):
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:
            tmp_path = f.name
        os.remove(tmp_path)

        try:
            log_feedback(
                "2026-01-01",
                "Õppeaine küsimus šžõäöü",
                "",
                [],
                ["Täpitähtedega aine"],
                "Võõrtähtedega vastus",
                "👍 Hea",
                "",
                file_path=tmp_path,
            )

            with open(tmp_path, 'r', encoding='utf-8') as f:
                content = f.read()
            assert "Õppeaine" in content
            assert "šžõäöü" in content
        finally:
            os.remove(tmp_path)


# ══════════════════════════════════════════════════════════════
# Testid: apply_filters
# ══════════════════════════════════════════════════════════════

class TestApplyFilters:
    def test_no_filters(self, sample_df):
        """Ilma filtriteta tagastatakse kõik read."""
        result = apply_filters(sample_df)
        assert len(result) == len(sample_df)

    def test_filter_by_semester(self, sample_df):
        result = apply_filters(sample_df, selected_semester=["kevad"])
        assert len(result) == 3
        assert all(result["semester"] == "kevad")

    def test_filter_by_keel(self, sample_df):
        result = apply_filters(sample_df, selected_keel=["inglise keel"])
        assert len(result) == 1
        assert result.iloc[0]["nimi_et"] == "Masinõpe"

    def test_filter_by_linn(self, sample_df):
        result = apply_filters(sample_df, selected_linn=["Tartu linn"])
        assert len(result) == 5  # kõik on Tartus

    def test_filter_by_nonexistent_linn(self, sample_df):
        result = apply_filters(sample_df, selected_linn=["Tallinn"])
        assert len(result) == 0

    def test_filter_by_oppeaste(self, sample_df):
        result = apply_filters(sample_df, selected_oppeaste=["magistriõpe"])
        # MTAT.03.263 ("bakalaureuseõpe, magistriõpe") ja LTAT.02.002 ("magistriõpe")
        assert len(result) == 2

    def test_filter_by_veebiope(self, sample_df):
        result = apply_filters(sample_df, selected_veebiope=["veebiõpe"])
        assert len(result) == 1
        assert result.iloc[0]["nimi_et"] == "Masinõpe"

    def test_filter_by_hindamisviis(self, sample_df):
        result = apply_filters(sample_df, selected_hindamisviis=["Eristamata (arv, m.arv, mi)"])
        assert len(result) == 1
        assert result.iloc[0]["nimi_et"] == "Saksa keel A1"

    def test_filter_by_eap_range(self, sample_df):
        result = apply_filters(sample_df, selected_eap=(3.0, 3.0))
        assert len(result) == 1
        assert result.iloc[0]["eap"] == 3.0

    def test_filter_by_eap_range_all(self, sample_df):
        result = apply_filters(sample_df, selected_eap=(0.0, 10.0))
        assert len(result) == 5

    def test_combined_filters(self, sample_df):
        """Mitu filtrit korraga."""
        result = apply_filters(
            sample_df,
            selected_semester=["kevad"],
            selected_keel=["eesti keel"],
        )
        # kevad + eesti keel: Andmeteaduse alused, Saksa keel A1, Sissejuhatus majandusteooriasse
        assert len(result) == 3

    def test_combined_strict_filters(self, sample_df):
        """Kitsad filtrid, mis annavad 0 tulemust."""
        result = apply_filters(
            sample_df,
            selected_semester=["sügis"],
            selected_keel=["inglise keel"],
            selected_oppeaste=["doktoriõpe"],
        )
        assert len(result) == 0

    def test_empty_dataframe(self):
        """Tühi DataFrame filtreeritakse ilma veata."""
        df = pd.DataFrame(columns=["semester", "keel", "linn", "oppeaste", "veebiope", "hindamisviis", "eap"])
        result = apply_filters(df, selected_semester=["kevad"])
        assert len(result) == 0


# ══════════════════════════════════════════════════════════════
# Testid: keyword_search
# ══════════════════════════════════════════════════════════════

class TestKeywordSearch:
    def test_finds_matching_keyword(self, sample_df):
        """Otsib võtmesõna, mis esineb ainult ühes aines."""
        result = keyword_search("tehisintellekt", sample_df)
        assert len(result) >= 1
        assert "Tehisintellekt I" in result["nimi_et"].values

    def test_finds_multiple_matches(self, sample_df):
        """Otsib sõna, mis esineb mitmes aines."""
        result = keyword_search("masinõpe", sample_df)
        assert len(result) >= 1

    def test_no_match(self, sample_df):
        """Otsing millegi kohta, mida ei eksisteeri."""
        result = keyword_search("kvantfüüsika", sample_df)
        assert len(result) == 0

    def test_case_insensitive(self, sample_df):
        """Otsing peab olema tõstutundetu."""
        result1 = keyword_search("Tehisintellekt", sample_df)
        result2 = keyword_search("tehisintellekt", sample_df)
        assert len(result1) == len(result2)

    def test_multi_word_query(self, sample_df):
        """Mitmesõnaline päring annab kõrgema skoori õigetele."""
        result = keyword_search("andmeteaduse alused", sample_df, top_k=5)
        if not result.empty:
            # Esimene tulemus peaks olema "Andmeteaduse alused"
            assert result.iloc[0]["nimi_et"] == "Andmeteaduse alused"

    def test_empty_query(self, sample_df):
        """Tühi päring ei tagasta tulemusi."""
        result = keyword_search("", sample_df)
        assert len(result) == 0

    def test_single_char_words_ignored(self, sample_df):
        """Ühetähelised sõnad jäetakse vahele."""
        result = keyword_search("a b c", sample_df)
        assert len(result) == 0

    def test_returns_keyword_score(self, sample_df):
        """Tulemusel on keyword_score veerg."""
        result = keyword_search("majandus", sample_df)
        if not result.empty:
            assert "keyword_score" in result.columns
            assert all(result["keyword_score"] > 0)

    def test_empty_dataframe(self):
        """Tühi DataFrame ei tekita viga."""
        df = pd.DataFrame(columns=["nimi_et", "nimi_en", "kirjeldus", "eesmargid", "opivaljundid"])
        result = keyword_search("test", df)
        assert len(result) == 0

    def test_top_k_limit(self, sample_df):
        """top_k piirab tulemuste arvu."""
        result = keyword_search("kursus", sample_df, top_k=2)
        assert len(result) <= 2


# ══════════════════════════════════════════════════════════════
# Testid: semantic_search_fn
# ══════════════════════════════════════════════════════════════

class TestSemanticSearch:
    def test_returns_results(self, sample_df, sample_embeddings):
        embedder = FakeEmbedder()
        results, ctx = semantic_search_fn(
            "tehisintellekt",
            sample_df,
            sample_embeddings,
            embedder,
            sample_df.index,
            top_k=3,
        )
        assert len(results) == 3
        assert "score" in results.columns
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_empty_indices(self, sample_df, sample_embeddings):
        embedder = FakeEmbedder()
        results, ctx = semantic_search_fn(
            "teste",
            sample_df,
            sample_embeddings,
            embedder,
            pd.Index([]),
            top_k=3,
        )
        assert results.empty
        assert "Ühtegi" in ctx

    def test_top_k_respected(self, sample_df, sample_embeddings):
        embedder = FakeEmbedder()
        results, _ = semantic_search_fn(
            "test",
            sample_df,
            sample_embeddings,
            embedder,
            sample_df.index,
            top_k=2,
        )
        assert len(results) == 2

    def test_scores_sorted_descending(self, sample_df, sample_embeddings):
        embedder = FakeEmbedder()
        results, _ = semantic_search_fn(
            "test",
            sample_df,
            sample_embeddings,
            embedder,
            sample_df.index,
            top_k=5,
        )
        scores = results["score"].tolist()
        assert scores == sorted(scores, reverse=True)


# ══════════════════════════════════════════════════════════════
# Testid: hybrid_search
# ══════════════════════════════════════════════════════════════

class TestHybridSearch:
    def test_returns_results(self, sample_df, sample_embeddings):
        embedder = FakeEmbedder()
        results, ctx = hybrid_search(
            "tehisintellekt",
            sample_df,
            sample_embeddings,
            embedder,
            sample_df.index,
            top_k=3,
        )
        assert len(results) <= 3
        assert "score" in results.columns
        assert "sem_score" in results.columns
        assert "kw_score" in results.columns

    def test_empty_indices(self, sample_df, sample_embeddings):
        embedder = FakeEmbedder()
        results, ctx = hybrid_search(
            "test",
            sample_df,
            sample_embeddings,
            embedder,
            pd.Index([]),
        )
        assert results.empty
        assert "Ühtegi" in ctx

    def test_keyword_weight_boosts_exact_match(self, sample_df, sample_embeddings):
        """Suurem keyword_weight peaks eelistama täpset tekstivastet."""
        embedder = FakeEmbedder()
        results_kw, _ = hybrid_search(
            "saksa keel",
            sample_df,
            sample_embeddings,
            embedder,
            sample_df.index,
            top_k=5,
            semantic_weight=0.0,
            keyword_weight=1.0,
        )
        if not results_kw.empty:
            assert results_kw.iloc[0]["nimi_et"] == "Saksa keel A1"

    def test_semantic_only(self, sample_df, sample_embeddings):
        """semantic_weight=1.0 peaks andma sama tulemuse kui puhas semantiline otsing."""
        embedder = FakeEmbedder()
        results, _ = hybrid_search(
            "test",
            sample_df,
            sample_embeddings,
            embedder,
            sample_df.index,
            top_k=5,
            semantic_weight=1.0,
            keyword_weight=0.0,
        )
        assert len(results) == 5

    def test_combined_weights(self, sample_df, sample_embeddings):
        """Vaikimisi kaalud (0.7/0.3) annavad tulemusi."""
        embedder = FakeEmbedder()
        results, ctx = hybrid_search(
            "andmeteadus masinõpe",
            sample_df,
            sample_embeddings,
            embedder,
            sample_df.index,
            top_k=5,
            semantic_weight=0.7,
            keyword_weight=0.3,
        )
        assert len(results) >= 1
        assert len(ctx) > 0


# ══════════════════════════════════════════════════════════════
# Testid: build_system_prompt
# ══════════════════════════════════════════════════════════════

class TestBuildSystemPrompt:
    def test_prompt_contains_context(self):
        prompt = build_system_prompt("Aine: Tehisintellekt I", 100)
        assert "Tehisintellekt I" in prompt
        assert "100" in prompt

    def test_prompt_no_filters(self):
        prompt = build_system_prompt("kontekst", 50)
        assert "Filtreid pole rakendatud" in prompt

    def test_prompt_with_semester_filter(self):
        prompt = build_system_prompt("kontekst", 10, selected_semester=["kevad"])
        assert "kevad" in prompt
        assert "Semester" in prompt

    def test_prompt_with_multiple_filters(self):
        prompt = build_system_prompt(
            "kontekst",
            5,
            selected_semester=["sügis"],
            selected_keel=["inglise keel"],
            selected_linn=["Tartu linn"],
        )
        assert "sügis" in prompt
        assert "inglise keel" in prompt
        assert "Tartu linn" in prompt

    def test_prompt_with_eap_filter(self):
        prompt = build_system_prompt(
            "kontekst",
            5,
            selected_eap=(3.0, 6.0),
            eap_min_val=0.0,
            eap_max_val=60.0,
        )
        assert "EAP" in prompt
        assert "3.0" in prompt

    def test_prompt_eap_not_shown_when_default(self):
        """EAP filtrit ei näidata, kui see on vaikeväärtus."""
        prompt = build_system_prompt(
            "kontekst",
            5,
            selected_eap=(0.0, 60.0),
            eap_min_val=0.0,
            eap_max_val=60.0,
        )
        # EAP filtrit ei peaks mainima
        lines = prompt.split("\n")
        eap_filter_lines = [l for l in lines if l.strip().startswith("EAP:")]
        assert len(eap_filter_lines) == 0

    def test_prompt_contains_instructions(self):
        prompt = build_system_prompt("kontekst", 10)
        assert "JUHISED" in prompt
        assert "eesti keeles" in prompt
        assert "hallutsin" in prompt.lower()

    def test_prompt_mentions_bot_name(self):
        prompt = build_system_prompt("kontekst", 10)
        assert "TÜ Kursuse Nõustaja" in prompt

    def test_prompt_top_k(self):
        prompt = build_system_prompt("kontekst", 10, top_k=3)
        assert "3" in prompt


# ══════════════════════════════════════════════════════════════
# Testid: calculate_cost
# ══════════════════════════════════════════════════════════════

class TestCalculateCost:
    def test_zero_tokens(self):
        cost = calculate_cost(0, 0)
        assert cost["input_cost"] == 0.0
        assert cost["output_cost"] == 0.0
        assert cost["total_cost"] == 0.0

    def test_known_values(self):
        # 1M sisendit * $0.10/1M = $0.10
        cost = calculate_cost(1_000_000, 0, cost_per_1m_input=0.10, cost_per_1m_output=0.20)
        assert abs(cost["input_cost"] - 0.10) < 1e-10
        assert cost["output_cost"] == 0.0
        assert abs(cost["total_cost"] - 0.10) < 1e-10

    def test_output_cost(self):
        cost = calculate_cost(0, 1_000_000, cost_per_1m_input=0.10, cost_per_1m_output=0.20)
        assert abs(cost["output_cost"] - 0.20) < 1e-10

    def test_combined_cost(self):
        cost = calculate_cost(500_000, 250_000, cost_per_1m_input=0.10, cost_per_1m_output=0.20)
        expected = 500_000 * 0.10 / 1_000_000 + 250_000 * 0.20 / 1_000_000
        assert abs(cost["total_cost"] - expected) < 1e-10

    def test_returns_dict_with_keys(self):
        cost = calculate_cost(100, 200)
        assert "input_cost" in cost
        assert "output_cost" in cost
        assert "total_cost" in cost

    def test_small_token_count(self):
        cost = calculate_cost(100, 50)
        assert cost["total_cost"] > 0
        assert cost["total_cost"] < 0.001  # väga väike kulu


# ══════════════════════════════════════════════════════════════
# Integratsioonitestid
# ══════════════════════════════════════════════════════════════

class TestIntegration:
    def test_filter_then_search(self, sample_df, sample_embeddings):
        """Integratsioon: filtreeri → otsi → ehita prompt."""
        # 1. Filtreeri
        filtered = apply_filters(sample_df, selected_semester=["sügis"])
        assert len(filtered) == 2

        # 2. Hübriidotsing
        embedder = FakeEmbedder()
        results, ctx = hybrid_search(
            "tehisintellekt",
            sample_df,
            sample_embeddings,
            embedder,
            filtered.index,
            top_k=2,
        )
        assert len(results) <= 2

        # 3. Ehita prompt
        prompt = build_system_prompt(
            ctx,
            n_filtered=len(filtered),
            selected_semester=["sügis"],
        )
        assert "sügis" in prompt
        assert str(len(filtered)) in prompt

    def test_full_pipeline_no_results(self, sample_df, sample_embeddings):
        """Integratsioon: filtrid, mis ei anna tulemusi."""
        filtered = apply_filters(sample_df, selected_linn=["Tallinn"])
        assert len(filtered) == 0

        embedder = FakeEmbedder()
        results, ctx = hybrid_search(
            "test",
            sample_df,
            sample_embeddings,
            embedder,
            filtered.index,
        )
        assert results.empty
        assert "Ühtegi" in ctx

    def test_cost_after_token_count(self):
        """Integratsioon: sõnumite tokenite arv → kulu arvutamine."""
        msgs = [
            {"role": "system", "content": "Oled assistent."},
            {"role": "user", "content": "Soovitan kursusi tehisintellekti kohta."},
        ]
        token_count = messages_to_token_count(msgs)
        assert token_count > 0

        cost = calculate_cost(token_count, 50)
        assert cost["total_cost"] > 0
