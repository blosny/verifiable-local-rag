import streamlit as st
import os
import sys

# Ana klasörü sys.path'e ekle
sys.path.insert(0, os.path.dirname(__file__))

from src.database import init_db, save_chunks, search_similar_chunks, clear_db
from src.ingest import process_document
from src.llm import LLMEngine
from src.verifier import verify_citations

# Sayfa Ayarları
st.set_page_config(
    page_title="Verifiable Local RAG",
    page_icon="🛡️",
    layout="wide"
)

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
    st.title("🛡️ Verifiable Local RAG")
    st.caption("Offline, Doğrulanabilir ve Alıntı Destekli Yerel RAG Platformu")
    st.markdown("---")
    
    st.subheader("📂 Doküman Yükle")
    uploaded_file = st.file_uploader("PDF veya TXT belgesi yükleyin", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        file_path = os.path.join(data_dir, uploaded_file.name)
        os.makedirs(data_dir, exist_ok=True)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner("Doküman parçalanıyor ve vektörleri veritabanına işleniyor..."):
                # 1. Dokümanı parçala (Page-Aware Overlapping Chunking)
                chunks = process_document(file_path)
                
                # 2. Vektörlerini üret
                for chunk in chunks:
                    chunk["embedding"] = llm_engine.generate_embedding(chunk["content"])
                    
                # 3. SQLite'a JSON String olarak yaz
                save_chunks(chunks)
                st.session_state.uploaded_files.append(uploaded_file.name)
                st.success(f"'{uploaded_file.name}' başarıyla işlendi! ({len(chunks)} parça eklendi)")
                
    st.markdown("---")
    st.subheader("📚 Yüklü Belgeler")
    if st.session_state.uploaded_files:
        for f in st.session_state.uploaded_files:
            st.text(f"• {f}")
    else:
        st.info("Henüz belge yüklenmedi.")
        
    st.markdown("---")
    if st.button("🗑️ Veritabanını Temizle", use_container_width=True):
        clear_db()
        st.session_state.uploaded_files = []
        st.session_state.messages = []
        st.success("Veritabanı ve sohbet geçmişi temizlendi!")
        st.rerun()

# --- ANA EKRAN ---
st.title("💬 Doğrulanabilir Yerel Yapay Zeka Asistanı")
st.markdown("Yüklediğiniz dokümanlara dayalı, **uydurma içermeyen (hallucination-free)** ve **sayfa alıntılı** yanıtlar üretir.")

tab1, tab2 = st.tabs(["💬 Asistan & Alıntılar", "🔍 Veritabanı & Vektör İnceleyici"])

with tab1:
    # Sohbet Geçmişi Gösterimi
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "verification" in msg and msg["verification"]:
                v = msg["verification"]
                st.markdown("---")
                st.caption(f"🛡️ **Doğrulama Skoru:** %{v['confidence_score']} ({v['verification_status']})")
                if v["verified_citations"]:
                    st.caption("📌 **Doğrulanan Alıntılar:** " + ", ".join([f"`{c}`" for c in v["verified_citations"]]))

    # Kullanıcı Soru Girişi
    if prompt := st.chat_input("Yüklediğiniz belgeler hakkında bir soru sorun..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Belgeler taranıyor ve alıntılar doğrulanıyor..."):
                # 1. Sorgunun vektörünü al
                query_vec = llm_engine.generate_embedding(prompt)
                
                # 2. SQLite veritabanından Kosinüs Benzerliği en yüksek kayıtları getir
                retrieved_chunks = search_similar_chunks(query_vec, top_k=3)
                
                # 3. LLM ile yanıt üret
                response_text = llm_engine.generate_answer(prompt, retrieved_chunks)
                
                # 4. Alıntı Doğrulama (Fact-Checker) çalıştır
                verification = verify_citations(response_text, retrieved_chunks)
                
                st.markdown(response_text)
                
                # Alıntı & Doğrulama Paneli
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Alıntı Doğruluk Skoru", f"%{verification['confidence_score']}")
                with col2:
                    st.metric("Doğrulama Durumu", verification['verification_status'])
                    
                if verification["verified_citations"]:
                    st.markdown("**📌 Doğrulanan Kaynaklar:**")
                    for cit in verification["verified_citations"]:
                        st.info(f"📄 {cit}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response_text,
            "verification": verification
        })

with tab2:
    st.subheader("🔍 SQLite Vektör Mağazası ve Benzerlik Skorları")
    st.caption("Bu sekme RAG arama mekanizmasının arka planda nasıl çalıştığını şeffaf bir şekilde görmenizi sağlar.")
    
    debug_query = st.text_input("Arama simülasyonu yapmak için bir kelime/cümle yazın:")
    if debug_query:
        q_vec = llm_engine.generate_embedding(debug_query)
        results = search_similar_chunks(q_vec, top_k=5)
        
        st.write(f"**'{debug_query}' sorgusu için en alakalı 5 veritabanı kaydı:**")
        for res in results:
            with st.expander(f"📄 {res['source_file']} (Sayfa {res['page_number']}) - Benzerlik Skoru: %{round(res['similarity_score']*100, 2)}"):
                st.write(res["content"])
                st.caption(f"Chunk Index: {res['chunk_index']} | DB ID: {res['id']}")
