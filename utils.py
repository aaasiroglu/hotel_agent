import requests
from PIL import Image
from io import BytesIO
import base64
import json
import torch
from services import clip_model, clip_preprocess, get_shoe_collection
import config
from openai import AzureOpenAI

# --- Azure OpenAI Client ---
try:
    azure_client = AzureOpenAI(
        api_key=config.openai_key,
        api_version="2024-02-15-preview",
        azure_endpoint=config.openai_endpoint,
        default_headers={"Ocp-Apim-Subscription-Key": config.openai_key}
    )
except Exception as e:
    print(f"ERROR: Could not initialize Azure OpenAI client: {e}")
    azure_client = None

# --- Utility Functions ---

async def extract_shoe_intent(user_query):
    """
    Analiz AI - Kullanıcı isteğini analiz eden ve arama parametrelerine dönüştüren fonksiyon.
    Akıcı ve kusursuz e-ticaret deneyimi için kritik bileşen.
    """
    if not azure_client:
        print("Azure OpenAI client not initialized, using default query extraction")
        return {"query": user_query}
    
    try:
        system_content = """Sen premium bir ayakkabı e-ticaret sitesinin analiz AI'ısın.
        Görevin: Kullanıcının doğal dildeki ayakkabı talebini analiz etmek ve onu verimli arama parametrelerine dönüştürmek.
        
        Örnek kullanıcı talepleri:
        - "Yarın akşam bir düğün var ve kırmızı elbisemin altına giyebileceğim şık bir topuklu ayakkabı arıyorum"
        - "Ofisin yanındaki gym'e gitmek için hem işte giyebileceğim hem de sporda kullanabileceğim rahat bir ayakkabı"
        - "Kış geliyor, su geçirmeyen bot önerebilir misiniz?"
        
        Her talep için şu bilgileri MUTLAKA çıkarmalısın:
        - Ayakkabı tipi (topuklu, spor, bot vs.)
        - Renk tercihleri
        - Stil özellikleri (şık, rahat, formal vs.)
        - Kullanım alanı (iş, spor, günlük vs.)
        - Özel istekler (su geçirmez, taşlı vs.)
        
        Cevabı şu formatta MUTLAKA JSON olarak döndür:
        {
          "query": "Kullanıcının isteğini karşılayacak optimize edilmiş arama sorgusu",
          "shoe_type": "Ayakkabı tipi",
          "color": "Renk tercihi veya null",
          "style": "Stil özellikleri veya null",
          "use_case": "Kullanım alanı veya null",
          "special_features": ["Özel istek 1", "Özel istek 2"] veya []
        }
        
        Sadece talebe cevap ver, konuşma başlatma, açıklama veya soru sorma.
        """
        
        response = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": f"Müşteri talebi: {user_query}"}
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content)
        print(f"🔍 Intent Analizi: {result}")
        return result
        
    except Exception as e:
        print(f"Intent analiz hatası: {e}")
        # Fallback: Direkt sorguyu kullan
        return {"query": user_query}

def get_base64_image_from_url(image_url):
    """Downloads an image from URL and converts it to base64."""
    try:
        response_img = requests.get(image_url, timeout=10)
        response_img.raise_for_status()
        img_bytes = response_img.content
        with Image.open(BytesIO(img_bytes)) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')
            return f"data:image/jpeg;base64,{base64_image}"
    except Exception as e:
        print(f"Error processing image {image_url} for Base64 encoding: {e}")
        return None

def get_clip_image_embedding(image_url):
    response = requests.get(image_url, timeout=10)
    response.raise_for_status()
    image = Image.open(BytesIO(response.content)).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0)
    with torch.no_grad():
        embedding = clip_model.encode_image(image_input)
        embedding = embedding.cpu().numpy().flatten().tolist()
    return embedding

def get_clip_text_embedding(text):
    import clip
    text_input = clip.tokenize([text])
    with torch.no_grad():
        embedding = clip_model.encode_text(text_input)
        embedding = embedding.cpu().numpy().flatten().tolist()
    return embedding


def insert_to_vector_db(image_url, json_data, description):
    """
    Inserts image metadata and a descriptive embedding into ChromaDB.
    """
    collection = get_shoe_collection()
    image_embedding = get_clip_image_embedding(image_url)
    metadata = {**json_data, "image_url": image_url}
    collection.add(
        documents=[description],
        embeddings=[image_embedding],
        ids=[image_url],
        metadatas=[metadata]
    )

def process_images_to_json_and_insert(image_links):
    """
    Processes a list of image URLs, analyzes them, and inserts results into DB.
    This function should be run ONCE to populate your database.
    """
    print(f"Starting initial image processing for {len(image_links)} images...")
    for i, image_url in enumerate(image_links):
        print(f"Processing image {i+1}/{len(image_links)}: {image_url}")
        try:
            json_data, description = image_to_json_and_caption(image_url)
            if json_data and description:
                insert_to_vector_db(image_url, json_data, description)
            else:
                print(f"Skipping insertion for {image_url} due to empty or invalid data from analysis.")
        except Exception as e:
            print(f"Error processing image {image_url}: {e}")
    print("Finished initial database population.")


def vector_search_shoes(query, top_k=5):
    """
    🚀 STABLE PROVEN SHOE SEARCH - Based on working hotel system
    Performs reliable vector search on ChromaDB with proven architecture.
    """
    try:
        collection = get_shoe_collection()
        
        # Enhanced query for better CLIP matching
        enhanced_query = enrich_shoe_query(query)
        print(f"🔍 Search Query: '{query}' → Enhanced: '{enhanced_query}'")
        
        # CLIP text embedding
        query_embedding = get_clip_text_embedding(enhanced_query)
        
        # Single reliable ChromaDB query (proven method)
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k * 2,  # Get extra for filtering
            include=["metadatas", "documents", "distances"]
        )
        
        if not results["documents"] or not results["documents"][0]:
            return "🔍 Veritabanımızda bu kriterlere uygun ayakkabı bulunamadı. Farklı kelimeler deneyin (bot, topuklu, spor, formal)."
        
        # Format results using actual database data (RAG approach)
        formatted_results = []
        for i, (metadata, doc, distance) in enumerate(zip(
            results["metadatas"][0][:top_k], 
            results["documents"][0][:top_k], 
            results["distances"][0][:top_k]
        )):
            # Use ACTUAL database values only
            shoe_name = metadata.get("name", f"Ayakkabı {i+1}")
            if shoe_name == "Unknown" or not shoe_name:
                shoe_name = create_realistic_shoe_name(metadata)
            
            description = metadata.get("description", doc)
            if not description or description == "No description":
                description = doc
            
            # Build attributes from actual metadata
            attributes = build_shoe_attributes(metadata)
            
            # Distance-based relevance (0-100%)
            relevance = max(0, int((1 - distance) * 100))
            
            image_url = metadata.get("image_url", "")
            
            formatted_results.append(f"""## 🔥 {shoe_name}

**{description}**

🏷️ **{attributes}**

📊 Uyum: {relevance}% | 🎯 Distance: {distance:.3f}

![{shoe_name}]({image_url})

---""")
        
        success_msg = f"🎯 **{len(formatted_results)} ayakkabı bulundu!**\n\n"
        return success_msg + "\n\n".join(formatted_results)
        
    except Exception as e:
        print(f"💥 Vector search error: {e}")
        return "🔧 Arama sisteminde geçici bir sorun var. Lütfen tekrar deneyin."

def create_realistic_shoe_name(metadata):
    """Create realistic shoe name from metadata - simple approach."""
    brand = metadata.get("brand", "Premium")
    shoe_type = metadata.get("shoe_type", "Ayakkabı")
    color = metadata.get("color", "")
    
    parts = []
    if brand and brand != "Unknown":
        parts.append(brand)
    if color:
        parts.append(color.title())
    parts.append(shoe_type.title())
    
    return " ".join(parts)

def build_shoe_attributes(metadata):
    """Build attribute string from actual metadata."""
    attributes = []
    
    if metadata.get("shoe_type"):
        attributes.append(f"Tip: {metadata['shoe_type'].title()}")
    if metadata.get("color"):
        attributes.append(f"Renk: {metadata['color'].title()}")
    if metadata.get("brand"):
        attributes.append(f"Marka: {metadata['brand']}")
    if metadata.get("style"):
        attributes.append(f"Stil: {metadata['style'].title()}")
    if metadata.get("material"):
        attributes.append(f"Malzeme: {metadata['material'].title()}")
    
    return " | ".join(attributes) if attributes else "Premium Kalite"

def enrich_shoe_query(query):
    """Enhance shoe search query for better CLIP matching - proven approach."""
    query_lower = query.lower()
    
    # Turkish to English mapping for better CLIP understanding
    turkish_to_english = {
        'topuklu': 'high heels stiletto',
        'şık': 'elegant formal dress',
        'abiye': 'evening formal dress',
        'spor': 'athletic sneaker sport',
        'bot': 'boots ankle boot',
        'yazlık': 'summer light',
        'kışlık': 'winter warm',
        'rahat': 'comfortable casual',
        'formal': 'formal dress business',
        'iş': 'business formal office',
        'günlük': 'casual daily wear',
        'yeşil': 'green',
        'siyah': 'black',
        'beyaz': 'white',
        'kahverengi': 'brown',
        'kırmızı': 'red',
        'mavi': 'blue',
        'taşlı': 'embellished rhinestone crystal',
        'parlak': 'shiny glittery metallic'
    }
    
    enhanced_terms = []
    for turkish, english in turkish_to_english.items():
        if turkish in query_lower:
            enhanced_terms.append(english)
    
    if enhanced_terms:
        return f"{query} {' '.join(enhanced_terms)}"
    
    return query

def filter_shoes_by_magnetic_relevance(original_query, results, top_k):
    """🎯 Ultra-intelligent relevance filtering with magnetic precision."""
    query_lower = original_query.lower()
    filtered = []
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0]
    )):
        # 🧠 Calculate multi-dimensional relevance score
        relevance_score = calculate_magnetic_relevance_score(query_lower, metadata)
        
        # 🎯 Distance boost (closer = better)
        distance_score = max(0, 1 - distance)
        
        # 🎪 Combine scores with smart weighting
        final_score = (relevance_score * 0.7) + (distance_score * 0.3)
        
        # 🚀 Only include high-quality matches
        if final_score > 0.1:  # Lower threshold for more results
            filtered.append({
                'metadata': metadata,
                'document': doc,
                'distance': distance,
                'relevance': final_score,
                'position': i
            })
    
    # 🎯 Sort by relevance score (descending)
    filtered.sort(key=lambda x: x['relevance'], reverse=True)
    
    return filtered[:top_k * 2]  # Return more for better selection

def calculate_magnetic_relevance_score(query_lower, metadata):
    """🧠 Advanced AI-powered relevance scoring."""
    score = 0.0
    
    # Extract metadata fields
    shoe_type = metadata.get("shoe_type", "").lower()
    color = metadata.get("color", "").lower()
    style = metadata.get("style", "").lower()
    material = metadata.get("material", "").lower()
    name = metadata.get("name", "").lower()
    description = metadata.get("description", "").lower()
    
    # 🎯 Primary type matching (highest priority)
    type_matches = {
        'topuklu': ['topuklu', 'heel', 'high'],
        'spor': ['spor', 'athletic', 'sneaker', 'sport'],
        'formal': ['formal', 'dress', 'business', 'elegant'],
        'bot': ['bot', 'boot', 'ankle'],
        'günlük': ['casual', 'daily', 'günlük'],
        'abiye': ['evening', 'formal', 'elegant', 'abiye']
    }
    
    for query_term, shoe_terms in type_matches.items():
        if query_term in query_lower:
            if any(term in shoe_type for term in shoe_terms):
                score += 0.5  # Strong type match
            if any(term in name for term in shoe_terms):
                score += 0.3  # Name type match
            if any(term in description for term in shoe_terms):
                score += 0.2  # Description type match
    
    # 🌈 Color matching
    color_matches = {
        'siyah': ['black', 'siyah'],
        'beyaz': ['white', 'beyaz'],
        'kahverengi': ['brown', 'kahverengi'],
        'kırmızı': ['red', 'kırmızı'],
        'mavi': ['blue', 'mavi'],
        'yeşil': ['green', 'yeşil']
    }
    
    for query_color, shoe_colors in color_matches.items():
        if query_color in query_lower:
            if any(c in color for c in shoe_colors):
                score += 0.3  # Perfect color match
            if any(c in name for c in shoe_colors):
                score += 0.2  # Name color match
    
    # ✨ Style and special features
    style_matches = {
        'şık': ['elegant', 'formal', 'chic', 'şık'],
        'rahat': ['comfortable', 'casual', 'rahat'],
        'taşlı': ['embellished', 'rhinestone', 'crystal', 'taşlı'],
        'parlak': ['shiny', 'glittery', 'metallic', 'parlak']
    }
    
    for query_style, shoe_styles in style_matches.items():
        if query_style in query_lower:
            if any(s in style for s in shoe_styles):
                score += 0.2
            if any(s in description for s in shoe_styles):
                score += 0.15
    
    # 🏢 Brand recognition bonus
    if any(brand in query_lower for brand in ['nike', 'adidas', 'lacoste']):
        brand_name = metadata.get("brand", "").lower()
        for brand in ['nike', 'adidas', 'lacoste']:
            if brand in query_lower and brand in brand_name:
                score += 0.1
    
    # 📝 Text similarity bonus
    query_words = query_lower.split()
    text_fields = [name, description, shoe_type, style]
    for word in query_words:
        if len(word) > 2:  # Skip short words
            for field in text_fields:
                if word in field:
                    score += 0.05
    
    return min(score, 1.0)  # Cap at 1.0

def create_magnetic_shoe_name(metadata):
    """🏷️ Create ultra-attractive shoe names from metadata."""
    name = metadata.get("name", "")
    if name and name != "Unknown":
        return name
    
    # 🎨 Smart name generation from metadata
    color = metadata.get("color", "")
    shoe_type = metadata.get("shoe_type", "")
    brand = metadata.get("brand", "")
    style = metadata.get("style", "")
    
    # Turkish name construction
    parts = []
    
    if brand and brand.lower() != "unknown":
        parts.append(brand)
    
    if color:
        parts.append(color.title())
    
    if style and style != shoe_type:
        parts.append(style.title())
    
    if shoe_type:
        parts.append(shoe_type.title())
    else:
        parts.append("Ayakkabı")
    
    return " ".join(parts) if parts else "Premium Ayakkabı"

def create_compelling_description(metadata, query):
    """📝 Create compelling descriptions that match user intent."""
    description = metadata.get("description", "")
    if description and description != "No description":
        return description
    
    # 🎯 Generate smart description based on query and metadata
    shoe_type = metadata.get("shoe_type", "ayakkabı")
    color = metadata.get("color", "")
    style = metadata.get("style", "")
    material = metadata.get("material", "")
    
    query_lower = query.lower()
    
    # Context-aware descriptions
    if "abiye" in query_lower or "şık" in query_lower:
        base = f"Özel günleriniz için tasarlanmış şık {shoe_type}"
    elif "spor" in query_lower or "rahat" in query_lower:
        base = f"Günlük konforunuz için ideal {shoe_type}"
    elif "iş" in query_lower or "formal" in query_lower:
        base = f"Profesyonel görünümünüz için mükemmel {shoe_type}"
    elif "bot" in query_lower:
        base = f"Dayanıklı ve tarz sahibi {shoe_type}"
    else:
        base = f"Kaliteli ve şık {shoe_type}"
    
    # Add color and material details
    details = []
    if color:
        details.append(f"{color} renkte")
    if material:
        details.append(f"{material} malzemesinde")
    
    if details:
        return f"{base}, {', '.join(details)} üretilmiştir."
    
    return f"{base}."

def get_key_attributes(metadata, query):
    """🌟 Extract and highlight key attributes based on query context."""
    attributes = []
    
    # Priority attributes based on query
    query_lower = query.lower()
    
    # Always show type and color
    if metadata.get("shoe_type"):
        attributes.append(f"📋 {metadata['shoe_type'].title()}")
    
    if metadata.get("color"):
        attributes.append(f"🎨 {metadata['color'].title()}")
    
    # Context-sensitive attributes
    if "marka" in query_lower or "brand" in query_lower:
        if metadata.get("brand"):
            attributes.append(f"🏷️ {metadata['brand']}")
    
    if metadata.get("style"):
        attributes.append(f"✨ {metadata['style'].title()}")
    
    if metadata.get("material"):
        attributes.append(f"🧵 {metadata['material'].title()}")
    
    # Special features highlight
    if "taşlı" in query_lower and "taşlı" in str(metadata.values()).lower():
        attributes.append("💎 Taşlı Detay")
    
    if "rahat" in query_lower:
        attributes.append("😌 Konforlu")
    
    return " | ".join(attributes) if attributes else "🔍 Premium Kalite"

def filter_shoes_by_relevance(original_query, results, top_k):
    """Filter shoes based on query relevance."""
    query_lower = original_query.lower()
    filtered = []
    
    for i, (doc, metadata, distance) in enumerate(zip(
        results["documents"][0], 
        results["metadatas"][0], 
        results["distances"][0]
    )):
        relevance_score = calculate_relevance_score(query_lower, metadata)
        
        if relevance_score > 0.3:  # Minimum relevance threshold
            filtered.append({
                'metadata': metadata,
                'document': doc,
                'distance': distance,
                'relevance': relevance_score
            })
    
    # Sort by relevance score (descending)
    filtered.sort(key=lambda x: x['relevance'], reverse=True)
    
    return filtered[:top_k]

def calculate_relevance_score(query_lower, metadata):
    """Calculate relevance score based on metadata match."""
    score = 0.0
    
    # Extract metadata fields
    shoe_type = metadata.get("shoe_type", "").lower()
    color = metadata.get("color", "").lower()
    style = metadata.get("style", "").lower()
    material = metadata.get("material", "").lower()
    
    # Type matching (highest priority)
    if any(term in query_lower for term in ['topuklu', 'topuk']) and 'topuklu' in shoe_type:
        score += 1.0
    elif any(term in query_lower for term in ['spor', 'athletic']) and 'spor' in shoe_type:
        score += 1.0
    elif any(term in query_lower for term in ['formal', 'iş']) and 'formal' in shoe_type:
        score += 1.0
    elif any(term in query_lower for term in ['bot', 'boot']) and 'bot' in shoe_type:
        score += 1.0
    
    # Color matching
    color_terms = ['siyah', 'beyaz', 'kahverengi', 'yeşil', 'kırmızı', 'mavi']
    for color_term in color_terms:
        if color_term in query_lower and color_term in color:
            score += 0.5
    
    # Style matching
    if any(term in query_lower for term in ['şık', 'abiye', 'elegant']) and any(s in style for s in ['şık', 'elegant']):
        score += 0.5
    
    # Material/feature matching
    if 'taşlı' in query_lower and any(term in material for term in ['taşlı', 'embellished']):
        score += 0.3
    
    return score

def create_turkish_shoe_name(metadata):
    """Create realistic Turkish shoe name."""
    brand = metadata.get("brand", "Premium")
    shoe_type = metadata.get("shoe_type", "Ayakkabı")
    color = metadata.get("color", "")
    
    if color:
        return f"{brand} {color.title()} {shoe_type.title()}"
    else:
        return f"{brand} {shoe_type.title()}"

def create_turkish_description(metadata, original_query):
    """Create contextual Turkish description."""
    shoe_type = metadata.get("shoe_type", "ayakkabı")
    color = metadata.get("color", "")
    material = metadata.get("material", "premium")
    style = metadata.get("style", "şık")
    
    query_lower = original_query.lower()
    
    # Context-aware descriptions
    if 'abiye' in query_lower or 'şık' in query_lower:
        if 'topuklu' in shoe_type:
            return f"Özel günler ve abiye kıyafetler için tasarlanmış {color} {shoe_type}. Şık görünümü ve rahat kullanımı ile gece boyu konforunuz için ideal."
        else:
            return f"Zarif ve sofistike tasarımıyla öne çıkan {color} {shoe_type}. Özel davetler için mükemmel bir seçim."
    
    elif 'iş' in query_lower or 'formal' in query_lower:
        return f"İş hayatı ve formal ortamlar için tasarlanmış {color} {shoe_type}. Profesyonel görünüm ve tüm gün konfor sağlar."
    
    elif 'spor' in query_lower or 'rahat' in query_lower:
        return f"Aktif yaşam tarzı için tasarlanmış {color} {shoe_type}. Üstün konfor ve dayanıklılık sunar."
    
    else:
        return f"Kaliteli {material} malzemeden üretilen {color} {shoe_type}. {style.title()} tasarımı ile her ortama uygun."

def create_turkish_attributes(metadata):
    """Create Turkish attributes list."""
    attributes = []
    
    if metadata.get("color"):
        attributes.append(f"Renk: {metadata['color'].title()}")
    
    if metadata.get("material"):
        material_tr = {
            'leather': 'Deri',
            'synthetic': 'Sentetik',
            'canvas': 'Kanvas',
            'suede': 'Süet'
        }.get(metadata['material'].lower(), metadata['material'].title())
        attributes.append(f"Malzeme: {material_tr}")
    
    if metadata.get("style"):
        attributes.append(f"Stil: {metadata['style'].title()}")
    
    if metadata.get("comfort_level"):
        comfort_tr = {
            'high': 'Yüksek Konfor',
            'medium': 'Orta Konfor',
            'low': 'Temel Konfor'
        }.get(metadata['comfort_level'].lower(), 'Konforlu')
        attributes.append(comfort_tr)
    
    if metadata.get("waterproof"):
        attributes.append("Su Geçirmez")
    
    # Default attributes if none found
    if not attributes:
        attributes = ["Premium Kalite", "Rahat Kullanım", "Şık Tasarım"]
    
    return ", ".join(attributes)


# TODO: Bu fonksiyon ileride re-ranking ve prompt zenginleştirme için kullanılabilir.
def enrich_prompt(prompt):
    """Uses GPT-4o to expand a short user query into a more descriptive paragraph for better embeddings."""
    if azure_client is None:
        print("Azure OpenAI client not initialized. Cannot enrich prompt.")
        return prompt

    try:
        completion = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Expand the following shoe search intent into a more descriptive paragraph, focusing on shoe features, style, and usage. Do not add conversational elements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Prompt Enrichment Error] {e}")
        return prompt
    

def image_to_json_and_caption(image_url):
    """
    Tek bir GPT-4o Vision çağrısı ile hem JSON hem caption döndürür.
    """
    if azure_client is None:
        print("Azure OpenAI client not initialized. Cannot perform image analysis.")
        return None, None

    b64_image = get_base64_image_from_url(image_url)
    if not b64_image:
        return None, None

    messages = [
        {
            "role": "system",
            "content": """
You are an expert shoe consultant and image analyst for Shoes AI, an online shoe shopping platform. You will extract all the potential details for customers' shoe preferences and queries.
You will observe much more on all the criteria required, especially for the following questions:
- Athletic shoes for running and sports,
- Formal shoes for business and professional use,
- Casual shoes for daily wear,
- Boots for winter and outdoor activities,
- High heels and dress shoes for special occasions,
- Comfortable shoes with good arch support,
- Waterproof shoes for wet conditions,
- Lightweight shoes for travel,
- Trendy and fashionable shoes for style-conscious customers.
- I will attend to ceremony and special event and i have green dress, what shoes should i wear?
- I need comfortable shoes for all-day wear.
- I want trendy shoes that match my outfit.
- I have wide feet and need shoes with good support.
- Kenarları taşlı ayakkabılar arıyorum.
- Böyle ince bir topuklu ayakkabı arıyorum.
- Düğüne gideceğim, şık ve etkileyici yanlarında çivi gibi detayları olan ayakkabı arıyorum.
- Markası AŞIROĞLU olan ayakkabı arıyorum.

For a given shoe image, do the following:
1. Output a valid JSON object with these fields:
   {
     "shoe_type": "string (athletic, formal, casual, boots, heels, etc.)",
     "color": "string",
     "material": "string (leather, synthetic, canvas, etc.)",
     "style": "string (modern, classic, trendy, etc.)",
     "brand": "string (estimated brand or Premium)",
     "season": "string (summer, winter, all-season)",
     "gender": "string (men, women, unisex)",
     "comfort_level": "string (high, medium, low)",
     "waterproof": boolean,
     "price_range": "string (budget, mid-range, premium)",
     "image_url": "string (original image URL)",
     "description": "string (short description of the shoe that inculude all details)",
     "attributes": "string (comma-separated list of attributes like 'comfortable, stylish, durable')"
   }
2. Then, write a short, user-friendly full detailed description (1-2 sentences) of the shoe for a shopping website.
IMPORTANT: Output STRICTLY in this format:
{
  "json": { ... },
  "caption": "..."
}
DO NOT add any explanation or commentary.
"""
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Analyze this shoe image and output the required JSON and caption."},
                {"type": "image_url", "image_url": {"url": b64_image}}
            ]
        }
    ]

    try:
        response = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            max_tokens=800,
            temperature=0
        )
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        json_data = result.get("json", {})
        caption = result.get("caption", "No description available.")
        return json_data, caption
    except Exception as e:
        print(f"ERROR: Failed to parse JSON+caption for image: {image_url}. Error: {e}")
        return None, None

def get_current_weather(city):
    """
    Returns the current weather for the given city using AccuWeather API.
    Falls back to OpenWeatherMap API if AccuWeather fails.
    """
    try:
        # First try AccuWeather API
        if hasattr(config, 'accuwather_key') and config.accuwather_key:
            # 1. Şehri AccuWeather Location API ile bul
            location_url = f"http://dataservice.accuweather.com/locations/v1/cities/search?apikey={config.accuwather_key}&q={city}&language=tr-tr"
            loc_resp = requests.get(location_url, timeout=5)
            loc_resp.raise_for_status()
            locations = loc_resp.json()
            if locations:
                location_key = locations[0]["Key"]
                # 2. Hava durumu bilgisini çek
                weather_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}?apikey={config.accuwather_key}&language=tr-tr&details=true"
                weather_resp = requests.get(weather_url, timeout=5)
                weather_resp.raise_for_status()
                weather = weather_resp.json()[0]
                text = weather["WeatherText"]
                temp = weather["Temperature"]["Metric"]["Value"]
                feels = weather["RealFeelTemperature"]["Metric"]["Value"]
                return f"{city} için güncel hava durumu: {text}, sıcaklık: {temp}°C, hissedilen: {feels}°C."
        
        # Fallback to OpenWeatherMap API
        if hasattr(config, 'weather_api_key') and config.weather_api_key:
            url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={config.weather_api_key}&units=metric&lang=tr"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                temp = data["main"]["temp"]
                description = data["weather"][0]["description"]
                feels_like = data["main"]["feels_like"]
                humidity = data["main"]["humidity"]
                
                return f"{city} için güncel hava durumu:\n🌡️ Sıcaklık: {temp}°C (Hissedilen: {feels_like}°C)\n☁️ Durum: {description}\n💧 Nem: {humidity}%"
        
        return f"{city} için hava durumu bilgisi şu anda mevcut değil."
        
    except Exception as e:
        print(f"[Weather API Error] {e}")
        return f"{city} için hava durumu şu anda alınamıyor."