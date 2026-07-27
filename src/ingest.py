import os
from typing import List, Dict, Any
from pypdf import PdfReader

def extract_text_by_pages(pdf_path: str) -> List[Dict[str, Any]]:
    """
    PDF dosyasını sayfa sayfa okur ve metinleri sayfa numarasıyla çıkarır.
    
    Returns:
        List of dicts: [{"page_number": 1, "text": "Sayfa 1 içeriği..."}, ...]
    """
    reader = PdfReader(pdf_path)
    pages_data = []
    
    for idx, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        text = text.strip()
        if text:  # Boş sayfaları atla
            pages_data.append({
                "page_number": idx + 1,  # Sayfa numarası 1'den başlar
                "text": text
            })
            
    return pages_data

def chunk_text(text: str, chunk_size: int = 250, overlap: int = 30) -> List[str]:
    """
    Verilen metni kelime bazlı, çakışmalı (overlapping) parçalara böler.
    
    Args:
        text: Parçalanacak metin
        chunk_size: Bir parçadaki maksimum kelime sayısı
        overlap: Parçalar arasında çakışacak kelime sayısı
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
        
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
        
        # Bir sonraki parçaya geçerken 'overlap' kadar geriden başla
        start += (chunk_size - overlap)
        
    return chunks

def process_document(file_path: str, chunk_size: int = 250, overlap: int = 30) -> List[Dict[str, Any]]:
    """
    Ana İşleme Fonksiyonu: PDF veya TXT dosyasını alır, parçalar ve metadata ekler.
    
    Returns:
        List of dicts: [
            {
                "source_file": "ornek.pdf",
                "page_number": 1,
                "chunk_index": 1,
                "content": "Parçalanmış metin..."
            }, ...
        ]
    """
    filename = os.path.basename(file_path)
    chunks_with_metadata = []
    
    if file_path.lower().endswith(".pdf"):
        pages = extract_text_by_pages(file_path)
        for page in pages:
            page_chunks = chunk_text(page["text"], chunk_size=chunk_size, overlap=overlap)
            for idx, chunk_content in enumerate(page_chunks):
                chunks_with_metadata.append({
                    "source_file": filename,
                    "page_number": page["page_number"],
                    "chunk_index": idx + 1,
                    "content": chunk_content
                })
                
    elif file_path.lower().endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        text_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk_content in enumerate(text_chunks):
            chunks_with_metadata.append({
                "source_file": filename,
                "page_number": 1,  # TXT dosyaları varsayılan 1. sayfadır
                "chunk_index": idx + 1,
                "content": chunk_content
            })
            
    return chunks_with_metadata
