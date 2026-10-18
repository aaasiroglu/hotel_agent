# Hotel Finder AI

Doğal dille otel odası arayan, görsel anlamsal arama yapan bir sohbet asistanı.

- **Arayüz:** [Chainlit](https://chainlit.io) (`main2.py`)
- **Dil modeli:** Azure OpenAI, function calling ile iki araç çağırır: oda araması ve hava durumu (AccuWeather)
- **Görsel arama:** Oda fotoğrafları CLIP ile gömülür, ChromaDB'de saklanır (`services.py`, `utils.py`)
- **İndeksleme:** `index_hotel_images.py` görsel bağlantılarını işleyip yapılandırılmış JSON + vektör olarak kaydeder

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # anahtarları doldur
python index_hotel_images.py   # chroma_db/ klasörünü üretir
chainlit run main2.py
```

## Notlar

- Anahtarlar yalnızca `.env` dosyasından okunur (`config.py`). `.env` ve `chroma_db/` repoya girmez.
- Aynı mimarinin ayakkabı sürümü: [shoe_finder_ai](https://github.com/aaasiroglu/shoe_finder_ai).
