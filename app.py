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

# 🏛️ Doğal, Klasik Masaüstü Uygulaması Standart Sayfa Ayarları
st.set_page_config(
    page_title="Verifiable Local RAG",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Yerel veritabanı başlatma
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

# --- SOL PANEL (Sidebar) ---
with st.sidebar:
    st.title("Verifiable Local RAG")
    st.write("Çevrimdışı Doküman Analizi ve Alıntı Doğrulama")
    st.divider()
    
    st.subheader("Doküman Yükleme")
    uploaded_files_batch = st.file_uploader(
        "PDF veya TXT dosyası ekleyin:", 
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
                with st.spinner(f"İşleniyor: {file.name}..."):
                    chunks = process_document(file_path)
                    for chunk in chunks:
                        chunk["embedding"] = llm_engine.generate_embedding(chunk["content"])
                    save_chunks(chunks)
                    st.session_state.uploaded_files.append(file.name)
                    st.toast(f"Yüklendi: {file.name}")
                    
    st.divider()
    st.subheader("Arama Kapsamı")
    doc_options = ["Tüm Belgeler"] + st.session_state.uploaded_files
    selected_doc_filter = st.selectbox(
        "Aranacak Belge:",
        options=doc_options,
        index=0
    )
    
    st.divider()
    st.subheader("Yüklü Belgeler")
    if st.session_state.uploaded_files:
        for f in st.session_state.uploaded_files:
            st.text(f"• {f}")
    else:
        st.write("Henüz belge yüklenmedi.")
        
    st.divider()
    if st.button("Veritabanını ve Sohbeti Sıfırla", use_container_width=True):
        clear_db()
        st.session_state.uploaded_files = []
        st.session_state.messages = []
        st.rerun()

# --- ANA EKRAN ---
st.header("Doğrulanabilir Doküman Asistanı")
st.write("Yüklenen belgeler üzerinden kaynak alıntılı ve doğrulanmış bilgi sunar.")

tab1, tab2 = st.tabs(["Sohbet ve Alıntılar", "Veritabanı İnceleyici"])

with tab1:
    # Sohbet Geçmişi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "verification" in msg and msg["verification"]:
                v = msg["verification"]
                st.divider()
                st.caption(f"Doğrulama Skoru: %{v['confidence_score']} | Durum: {v['verification_status']}")
                if v["verified_citations"]:
                    st.caption("Kaynaklar: " + ", ".join([f"{c}" for c in v["verified_citations"]]))

    # Soru Girişi
    if prompt := st.chat_input("Sorunuzu yazın..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Yanıt ve alıntılar hazırlanıyor..."):
                query_vec = llm_engine.generate_embedding(prompt)
                
                retrieved_chunks = retrieve_smart_chunks(
                    query_text=prompt,
                    query_embedding=query_vec,
                    top_k=3,
                    filter_source=selected_doc_filter
                )
                
                response_text = llm_engine.generate_answer(prompt, retrieved_chunks)
                verification = verify_citations(response_text, retrieved_chunks)
                
                st.write(response_text)
                
                # Panel
                st.divider()
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Alıntı Doğruluk Skoru", f"%{verification['confidence_score']}")
                with col2:
                    st.metric("Doğrulama Durumu", verification['verification_status'])
                    
                if verification["verified_citations"]:
                    st.write("**Doğrulanan Kaynaklar:**")
                    for cit in verification["verified_citations"]:
                        st.info(f"Kaynak: {cit}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "verification": verification
        })

with tab2:
    st.subheader("SQLite Vektör Mağazası")
    st.write("Veritabanı kayıtlarını inceleyin.")
    
    debug_query = st.text_input("Arama simülasyonu için kelime girin:")
    if debug_query:
        q_vec = llm_engine.generate_embedding(debug_query)
        results = retrieve_smart_chunks(
            query_text=debug_query,
            query_embedding=q_vec,
            top_k=5,
            filter_source=selected_doc_filter
        )
        
        st.write(f"'{debug_query}' için en alakalı 5 parça:")
        for res in results:
            with st.expander(f"{res['source_file']} (Sayfa {res['page_number']}) - Skor: %{round(res['similarity_score']*100, 2)}"):
                st.write(res["content"])
                st.caption(f"Chunk Index: {res['chunk_index']} | DB ID: {res['id']}")
