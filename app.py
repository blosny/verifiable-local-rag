import streamlit as st
import os
import sys

# Ana klasörü sys.path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

from src.database import init_db, save_chunks, clear_db
from src.ingest import process_document
from src.retriever import retrieve_smart_chunks
from src.llm import LLMEngine
from src.verifier import verify_citations

# 🎨 Ultra Modern Dashboard (Vercel / GitHub Style Minimal Slate Theme)
st.set_page_config(
    page_title="Verifiable Local RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS Styling
st.markdown("""
<style>
    /* Global Reset & Slate Background */
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    
    /* Top Header Bar */
    .header-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        padding: 1.2rem 1.6rem;
        border-radius: 10px;
        margin-bottom: 1.5rem;
    }
    .header-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }
    .header-desc {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 0.2rem;
    }

    /* Sidebar Clean Layout */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1e293b;
    }

    /* File List Clean Badge */
    .file-badge {
        background-color: #1e293b;
        color: #38bdf8;
        padding: 0.35rem 0.6rem;
        border-radius: 6px;
        font-family: monospace;
        font-size: 0.8rem;
        margin-bottom: 0.4rem;
        border: 1px solid #334155;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block;
    }
    
    /* Custom Metric Display Card */
    .score-card {
        background-color: #111827;
        border: 1px solid #1e293b;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Veritabanı İlklendirme
init_db()

@st.cache_resource
def get_llm_engine():
    return LLMEngine()

llm_engine = get_llm_engine()

# Session State Yönetimi
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- SIDEBAR (Sol Panel) ---
with st.sidebar:
    st.markdown("### Doküman Yönetimi")
    st.caption("Çevrimdışı Vektör Mağazası")
    st.markdown("---")
    
    uploaded_files_batch = st.file_uploader(
        "PDF / TXT Belgesi Yükleyin", 
        type=["pdf", "txt"],
        accept_multiple_files=True
    )
    
    if uploaded_files_batch:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        for file in uploaded_files_batch:
            file_path = os.path.join(data_dir, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
                
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"'{file.name}' ayrıştırılıyor..."):
                    chunks = process_document(file_path)
                    for chunk in chunks:
                        chunk["embedding"] = llm_engine.generate_embedding(chunk["content"])
                    save_chunks(chunks)
                    st.session_state.uploaded_files.append(file.name)
                    st.toast(f"Yüklendi: {file.name}")
                    
    st.markdown("---")
    doc_options = ["Tüm Belgeler"] + st.session_state.uploaded_files
    selected_doc_filter = st.selectbox(
        "Arama Filtresi:",
        options=doc_options,
        index=0
    )
    
    st.markdown("---")
    st.markdown("##### İndekslenen Belgeler")
    if st.session_state.uploaded_files:
        for f in st.session_state.uploaded_files:
            st.markdown(f'<div class="file-badge" title="{f}">{f}</div>', unsafe_allow_html=True)
    else:
        st.info("İndekslenen belge yok.")
        
    st.markdown("---")
    if st.button("Veritabanını Sıfırla", use_container_width=True):
        clear_db()
        st.session_state.uploaded_files = []
        st.session_state.messages = []
        st.rerun()

# --- ANA EKRAN HEADER ---
st.markdown("""
<div class="header-card">
    <div class="header-title">Verifiable Local RAG Dashboard</div>
    <div class="header-desc">Yerel veritabanınızdaki belgeler üzerinden %100 doğrulanabilir ve kaynak alıntılı analiz üretir.</div>
</div>
""", unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Sohbet & Alıntı Paneli", "Vektör Mağazası İnceleyici"])

with tab1:
    # Geçmiş Sohbet Gösterimi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "verification" in msg and msg["verification"]:
                v = msg["verification"]
                st.markdown("---")
                st.caption(f"Doğrulama Skoru: **%{v['confidence_score']}** | Durum: **{v['verification_status']}**")
                if v["verified_citations"]:
                    st.caption("Doğrulanan Kaynaklar: " + ", ".join([f"`{c}`" for c in v["verified_citations"]]))

    # Soru Giriş Barı
    if prompt := st.chat_input("Belgeleriniz hakkında bir soru yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Vektör araması ve doğrulama yapılıyor..."):
                query_vec = llm_engine.generate_embedding(prompt)
                
                retrieved_chunks = retrieve_smart_chunks(
                    query_text=prompt,
                    query_embedding=query_vec,
                    top_k=3,
                    filter_source=selected_doc_filter
                )
                
                response_text = llm_engine.generate_answer(prompt, retrieved_chunks)
                verification = verify_citations(response_text, retrieved_chunks)
                
                st.markdown(response_text)
                
                # Doğrulama Skor Paneli
                st.markdown("---")
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Alıntı Doğruluk Skoru", f"%{verification['confidence_score']}")
                with c2:
                    st.metric("Doğrulama Durumu", verification['verification_status'])
                    
                if verification["verified_citations"]:
                    st.markdown("**Kaynak Alıntıları:**")
                    for cit in verification["verified_citations"]:
                        st.info(f"📄 {cit}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "verification": verification
        })

with tab2:
    st.subheader("SQLite Vektör Mağazası (Debug)")
    st.caption("Arama mekanizmasını şeffaf şekilde inceleyin.")
    
    debug_query = st.text_input("Vektör simülasyonu için kelime girin:")
    if debug_query:
        q_vec = llm_engine.generate_embedding(debug_query)
        results = retrieve_smart_chunks(
            query_text=debug_query,
            query_embedding=q_vec,
            top_k=5,
            filter_source=selected_doc_filter
        )
        
        st.write(f"**'{debug_query}' için getirilen en alakalı 5 parça:**")
        for res in results:
            with st.expander(f"{res['source_file']} (Sayfa {res['page_number']}) - Skor: %{round(res['similarity_score']*100, 2)}"):
                st.write(res["content"])
                st.caption(f"Chunk Index: {res['chunk_index']} | DB ID: {res['id']}")
