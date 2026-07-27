# Verifiable Local RAG

> Offline, Doğrulanabilir ve Alıntı Destekli Yerel RAG (Retrieval-Augmented Generation) Platformu.
> Microsoft Foundry Local SDK altyapısı kullanılarak geliştirilmiştir.

---

## 🏛️ Sistem Mimarisi

```text
+-----------------------------------------------------------------------------------+
|                                 1. ARAYÜZ (UI)                                    |
|                       (Streamlit - Doküman Analiz Dashboard)                      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                              2. VERİ İŞLEME PİPELİNE                              |
|  - PDF / TXT Okuyucu (pdfplumber ile Tablo Ayrıştırma)                             |
|  - Metin Parçalama (Page-Aware Overlapping Chunking)                              |
|  - Vektör Motoru (Foundry Local SDK - 384D Embeddings)                            |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            3. VEKTÖR VERİTABANI KATMANI                           |
|  - SQLite DB (Metin + Sayfa Metadata + JSON Vektör Dizileri)                      |
|  - Akıllı Getirici (Smart Retriever: Document Score Boosting)                      |
+-----------------------------------------+-----------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                        4. LLM VE ALINTI DOĞRULAMA MOTORU                          |
|  - Foundry Local LLM Runtime (Phi-3.5-mini / Qwen-2.5)                             |
|  - Fact-Checker Verifier (Cümle Düzeyinde Alıntı ve Doğrulama Skoru)              |
+-----------------------------------------------------------------------------------+
```

---

## 🚀 Öne Çıkan Özellikler

1. **%100 Çevrimdışı (Offline) Güvenlik:** Verileriniz internete çıkmaz, tüm vektörleştirmeler ve LLM yanıtları yerel bilgisayarda çalışır.
2. **Uydurmasız Yanıt Garantisi (Strict No-Hallucination):** Model sadece yüklenen belgelerdeki verilere göre yanıt verir.
3. **Cümle Düzeyinde Alıntı Doğrulama:** Üretilen her yanıtın belgedeki hangi sayfadan alındığı doğrulanır ve dürüstçe %0 - %100 Doğruluk Skoru verilir.
4. **PDF Tablo & Sayısal Veri Desteği (`pdfplumber`):** Tabloları hücre matrisi olarak okur ve LLM'in anlayacağı Markdown tablosuna çevirir.
5. **Akıllı Çoklu Doküman Filtreleme:** Sorgudaki ipuçlarını analiz ederek ilgili dokümana bonus benzerlik skoru verir.

---

## 💻 Kurulum ve Çalıştırma

### 1. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 2. Uygulamayı Başlatın
```bash
streamlit run app.py
```

---

## 🧪 Testleri Çalıştırma

Tüm birim (unit) testleri çalıştırmak için:

```bash
python tests/test_database.py
python tests/test_ingest.py
python tests/test_llm.py
python tests/test_verifier.py
python tests/test_retriever.py
```
