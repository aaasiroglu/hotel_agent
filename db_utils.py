from chromadb import PersistentClient

def get_chroma_client():
    """
    Initialize and return a ChromaDB PersistentClient.
    """
    try:
        client = PersistentClient(path="./chroma_db")
        return client
    except Exception as e:
        print(f"Error initializing ChromaDB client: {e}")
        return None

def get_hotel_collection(client):
    """
    Get the hotel collection from the ChromaDB client.
    If it doesn't exist, create it.
    """
    try:
        collection = client.get_or_create_collection(name="hotel_images")
        return collection
    except Exception as e:
        print(f"Error getting or creating hotel collection: {e}")
        return None

def print_all_records(collection):
    """
    Print all records in the collection with their ids, metadatas, and documents.
    """
    try:
        results = collection.get(include=["metadatas", "documents"])
        ids = results.get("ids", [])
        metadatas = results.get("metadatas", [])
        documents = results.get("documents", [])
        total = len(ids)
        if total == 0:
            print("No records found in the collection.")
            return
        print(f"Total records: {total}\n")
        for idx, (id_, metadata, doc) in enumerate(zip(ids, metadatas, documents)):
            print(f"--- Record {idx+1} ---")
            print(f"ID: {id_}")
            print(f"Metadata: {metadata}")
            print(f"Document: {doc}\n")
    except Exception as e:
        print(f"Error fetching records: {e}")

if __name__ == "__main__":
    client = get_chroma_client()
    collection = get_hotel_collection(client)
    if collection:
        print_all_records(collection)
    else:
        print("Collection could not be loaded.")
