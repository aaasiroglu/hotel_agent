==============================
Premium Shoes AI – README
==============================

📌 Project Overview:
This system allows users to search for shoes using natural language queries.
It leverages GPT-4o, CLIP-based embeddings, and ChromaDB for visual semantic search and ranking.

------------------------------
Setup & Execution Flow
------------------------------

1. Initialize Services and Models:
   - File: services.py
   - Purpose: Loads CLIP model and initializes ChromaDB connection.

2. Populate the Vector Database:
   - File: index_shoes.py
   - Description: Processes all shoe image URLs and stores their structured JSON + vector embeddings into ChromaDB.

3. (Optional) Verify VectorDB Content:
   - File: db_utils.py
   - Usage: Can be used to inspect or debug contents of the `shoe_images` collection in ChromaDB.

4. Launch the Application:
   - File: main2.py
   - Run with:
     ```bash
     python -m chainlit run main2.py
     ```
   - Description: Starts the chatbot interface where users can input shoe preferences and receive visual search results.

------------------------------
Project Structure Summary:
- services.py           → Loads CLIP model + ChromaDB client
- index_shoes.py        → One-time embedding + DB insertion script
- db_utils.py           → Utility to inspect vector DB
- main2.py              → Chatbot interface with GPT-4o + semantic tools
- utils.py              → All functions used in this project
	
--------------------------------------------------------
Author: Arif Asiroglu
Date: 21.7.2025 - Updated for Premium Shoes AI