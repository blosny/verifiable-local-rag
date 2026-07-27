import json
import sqlite3
import numpy as np
from typing import List, Dict, Any, Optional
from src.database import DB_PATH, cosine_similarity

def retrieve_smart_chunks(
    query_text: str,
    query_embedding: List[float],
    top_k: int = 3,
    filter_source: Optional[str] = None,
    db_path: str = DB_PATH
) -> List[Dict[str, Any]]:
    """
    Çoklu doküman ortamında akıllı arama ve filtreleme yapan gelişmiş Retriever.
    
    Args:
        query_text: Kullanıcının sorduğu ham metin
        query_embedding: Sorgunun 384 boyutlu vektörü
        top_k: Getirilecek maksimum parça sayısı
        filter_source: Belirli bir dosya adına göre filtreleme (Opsiyonel)
        db_path: SQLite veritabanı yolu
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Kullanıcı belirli bir dosya filtrelediyse SQL seviyesinde süz
    if filter_source and filter_source != "Tüm Belgeler":
        cursor.execute(
            "SELECT id, source_file, page_number, chunk_index, content, embedding FROM document_chunks WHERE source_file = ?",
            (filter_source,)
        )
    else:
        cursor.execute("SELECT id, source_file, page_number, chunk_index, content, embedding FROM document_chunks")
        
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        return []
        
    results = []
    query_words = set(query_text.lower().split())
    
    for row in rows:
        chunk_id, source_file, page_number, chunk_index, content, embedding_json = row
        chunk_vector = json.loads(embedding_json)
        
        # Vektör boyut uyuşmazlığı kontrolü
        if len(query_embedding) != len(chunk_vector):
            continue
            
        base_score = cosine_similarity(query_embedding, chunk_vector)
        
        # --- AKILLI BOOSTING ALGORİTMASI ---
        # 1. Dosya Adı İpucu Boosting: Eğer dosya adındaki kelimeler soruda geçiyorsa skorunu artır
        file_words = set(source_file.lower().replace("_", " ").replace(".", " ").split())
        if query_words.intersection(file_words):
            base_score += 0.15  # %15 Bonus Skor
            
        # 2. Sayfa 1 / Başlık Boosting: Dokümanların ilk sayfaları genelde tanım ve başlık içerir
        if page_number == 1 and ("nedir" in query_text.lower() or "ne demek" in query_text.lower()):
            base_score += 0.10  # %10 Tanım Bonusu
            
        results.append({
            "id": chunk_id,
            "source_file": source_file,
            "page_number": page_number,
            "chunk_index": chunk_index,
            "content": content,
            "similarity_score": min(base_score, 1.0) # Maksimum 1.0 skor sınırı
        })
        
    # En yüksek skordan en düşüğe sırala
    results.sort(key=lambda x: x["similarity_score"], reverse=True)
    return results[:top_k]
