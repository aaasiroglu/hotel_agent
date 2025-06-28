import clip
from chromadb import PersistentClient

# CLIP modelini ve preprocess fonksiyonunu bir kez yükle
clip_model, clip_preprocess = clip.load("ViT-B/32")

# ChromaDB client'ı bir kez başlat
chroma_client = PersistentClient(path="./chroma_db")

def get_hotel_collection():
    return chroma_client.get_or_create_collection(name="hotel_images")