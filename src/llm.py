import numpy as np
from typing import List, Dict, Any

# Microsoft Foundry Local SDK kontrolü
HAS_FOUNDRY_LOCAL = False
try:
    from foundry_local_sdk import FoundryLocalClient
    HAS_FOUNDRY_LOCAL = True
except ImportError:
    HAS_FOUNDRY_LOCAL = False

class LLMEngine:
    def __init__(self, model_name: str = "phi-3.5-mini"):
        self.model_name = model_name
        self.client = None
        
        # Eğer Foundry Local kütüphanesi mevcutsa istemciyi başlat
        if HAS_FOUNDRY_LOCAL:
            try:
                self.client = FoundryLocalClient()
            except Exception as e:
                print(f"Foundry Local ilklendirilemedi, Fallback mod aktif: {e}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Metin için 384 boyutlu vektör (embedding) üretir.
        Geriye float listesi döner (JSON serialization için uygun).
        """
        if self.client and hasattr(self.client, "embeddings"):
            try:
                res = self.client.embeddings.create(input=text, model=self.model_name)
                return res.data[0].embedding
            except Exception:
                pass
                
        # Fallback: Kütüphane yüklenemediğinde çalışan yerel determinist hash algoritması
        return self._simple_hash_embedding(text)

    def _simple_hash_embedding(self, text: str, dim: int = 384) -> List[float]:
        """Kütüphane bağımlılığı olmadan determinist vektör üreten fallback algoritması."""
        vec = np.zeros(dim, dtype=np.float32)
        words = text.lower().split()
        for word in words:
            idx = abs(hash(word)) % dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def generate_answer(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Veritabanından çekilen parçaları (context) kullanarak LLM yanıtı üretir.
        """
        if not context_chunks:
            return "Üzgünüm, yüklenen belgelerde bu soruyla ilgili yeterli bilgi bulunamadı."
            
        # Bağlamı (Context) düzenli bir formatta birleştir
        context_str = ""
        for idx, chunk in enumerate(context_chunks, 1):
            context_str += f"\n--- [KAYNAK {idx}: {chunk['source_file']} (Sayfa {chunk['page_number']})] ---\n"
            context_str += f"{chunk['content']}\n"
            
        # System Prompt Engineering: Yanılsamayı (Hallucination) engelleyen şablon
        prompt = f"""Sen doğrulanabilir bilgiler sunan dürüst bir Yapay Zeka Asistanısın.
Aşağıda verilen KAYNAK METİNLERİ dikkatlice oku ve SADECE bu metinlerdeki bilgileri kullanarak kullanıcının sorusunu yanıtla.
Eğer verilen metinlerde sorunun cevabı yoksa, bilmediğini ve kaynaklarda olmadığını dürüstçe belirt. ASLA uydurma bilgi verme.
Yanıt verirken bilgiyi aldığın kaynak belge adını ve sayfa numarasını belirt.

KAYNAK METİNLER:
{context_str}

KULLANICI SORUSU:
{query}

YANIT:"""

        # Yerel LLM İstemcisi Çalışıyorsa Modele Gönder
        if self.client and hasattr(self.client, "chat"):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Foundry Local LLM yanıtı alınamadı, Fallback yanıt üretiliyor: {e}")

        # Fallback Yanıt Oluşturucu (Local LLM indirilmeyen durumlar için mock/rule-based yanıt)
        top_chunk = context_chunks[0]
        return f"Belgelerde bulunan bilgilere göre: {top_chunk['content']}\n\n(Kaynak: {top_chunk['source_file']}, Sayfa {top_chunk['page_number']})"
