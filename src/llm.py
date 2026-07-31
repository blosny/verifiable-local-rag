import numpy as np
from typing import List, Dict, Any

HAS_FOUNDRY_LOCAL = False
try:
    from foundry_local_sdk import Configuration, FoundryLocalManager
    HAS_FOUNDRY_LOCAL = True
except ImportError:
    HAS_FOUNDRY_LOCAL = False

class LLMEngine:
    def __init__(self, model_id: str = "qwen2.5-0.5b-instruct-generic-cpu:4"):
        self.model_id = model_id
        self.client = None
        self.foundry_model = None
        self.is_foundry_active = False
        
        if HAS_FOUNDRY_LOCAL:
            try:
                config = Configuration("VerifiableLocalRAG")
                manager = FoundryLocalManager(config)
                
                # Execution Providers Kontrolü & İndirme
                try:
                    manager.download_and_register_eps()
                except Exception as ep_err:
                    print(f"EP indirme uyarısı: {ep_err}")
                
                # Kataloktaki tam ID ile modeli bul
                models = manager.catalog.list_models()
                target_model = None
                for m in models:
                    if m.id == self.model_id:
                        target_model = m
                        break
                        
                if target_model:
                    if not target_model.is_cached:
                        print(f"Foundry Local model indiriliyor: {self.model_id}...")
                        target_model.download()
                    target_model.load()
                    self.foundry_model = target_model
                    self.is_foundry_active = True
                    print(f"Foundry Local SDK aktif: {self.model_id}")
            except Exception as e:
                print(f"Foundry Local ilklendirilemedi, Fallback mod aktif: {e}")
                self.is_foundry_active = False

    def generate_embedding(self, text: str) -> List[float]:
        """
        Metin için 384 boyutlu vektör (embedding) üretir.
        """
        if self.is_foundry_active and self.foundry_model:
            try:
                emb_client = self.foundry_model.get_embedding_client()
                res = emb_client.create(input=text)
                return res.data[0].embedding
            except Exception:
                pass
                
        # Fallback: Yerel determinist hash algoritması
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
            return "Yüklenen belgelerde bu soruyla ilgili yeterli bilgi bulunmamaktadır."
            
        context_str = ""
        for idx, chunk in enumerate(context_chunks, 1):
            # Çirkin || boru işaretlerini temizle
            clean_content = chunk['content'].replace("|||---|---|---|---|", "").replace("||", " ").strip()
            context_str += f"\n--- [KAYNAK {idx}: {chunk['source_file']} (Sayfa {chunk['page_number']})] ---\n"
            context_str += f"{clean_content}\n"
            
        prompt = f"""Sen doğrulanabilir bilgiler sunan dürüst bir Yapay Zeka Asistanısın.
Aşağıda verilen KAYNAK METİNLERİ dikkatlice oku ve SADECE bu metinlerde geçen gerçek bilgileri kullanarak kullanıcının sorusunu okunaklı Türkçe cümlelerle yanıtla.

KATI KURALLAR:
1. Sorunun yanıtı verilen kaynak metinlerde açıkça geçmiyorsa kesinlikle 'Yüklenen belgelerde bu soruyla ilgili yeterli bilgi bulunmamaktadır.' de.
2. Metindeki ham boru '|' işaretlerini ve tablo taslaklarını olduğu gibi kopyalama, okunaklı cümlelere çevir!
3. Asla genel kültüründen, tahminlerinden veya dış kaynaklardan yanıt uydurma!

KAYNAK METİNLER:
{context_str}

KULLANICI SORUSU:
{query}

YANIT:"""

        if self.is_foundry_active and self.foundry_model:
            try:
                chat_client = self.foundry_model.get_chat_client()
                response = chat_client.complete_chat(
                    messages=[{"role": "user", "content": prompt}]
                )
                ans = response.choices[0].message.content
                # Yanıttaki çirkin boru sembollerini temizle
                ans = ans.replace("|||---|---|---|---|", "").replace("|||", "").replace("||", "")
                return ans
            except Exception as e:
                print(f"Foundry Local LLM yanıt hatası: {e}")

        # Fallback Yanıt Oluşturucu
        top_chunk = context_chunks[0]
        clean_top = top_chunk['content'].replace("|||---|---|---|---|", "").replace("||", " ").strip()
        return f"Belgelerde bulunan bilgilere göre: {clean_top}\n\n(Kaynak: {top_chunk['source_file']}, Sayfa {top_chunk['page_number']})"
