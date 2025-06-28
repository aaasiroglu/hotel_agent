import requests
from PIL import Image
from io import BytesIO
import base64
import json
import torch
from services import clip_model, clip_preprocess, get_hotel_collection
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
    collection = get_hotel_collection()
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

def vector_search_hotel_rooms(query, top_k=5):
    """
    Performs an initial vector search on the ChromaDB to find top_k relevant image URLs.
    """
    collection = get_hotel_collection()
    query_embedding = get_clip_text_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["metadatas", "documents", "distances"]
    )
    combined_results = [
        {
            "url": id_,
            "metadata": metadata,
            "description": doc,
            "distance": dist
        }
        for id_, metadata, doc, dist in zip(
            results["ids"][0], results["metadatas"][0], results["documents"][0], results["distances"][0]
        )
    ]
    return combined_results

def enrich_prompt(prompt):
    """Uses GPT-4o to expand a short user query into a more descriptive paragraph for better embeddings."""
    if azure_client is None:
        print("Azure OpenAI client not initialized. Cannot enrich prompt.")
        return prompt

    try:
        completion = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Expand the following hotel room search intent into a more descriptive paragraph, focusing on room features and amenities. Do not add conversational elements."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        print(f"[Prompt Enrichment Error] {e}")
        return prompt

def gpt4o_filter_images_with_reason(image_urls, user_prompt):
    """
    Takes a list of image URLs (e.g., top 15 from vector search).
    Sends them to GPT-4o Vision for detailed visual analysis and selection (up to 3-5 rooms).
    Returns selected image URLs and a comprehensive explanation.
    """
    if azure_client is None:
        return json.dumps({
            "selected": [],
            "explanation": "Azure OpenAI client not initialized. Cannot perform visual filtering."
        })

    images_to_send_to_gpt = []
    for i, image_url in enumerate(image_urls):
        b64_image = get_base64_image_from_url(image_url)
        if b64_image:
            images_to_send_to_gpt.append({
                "type": "image_url",
                "image_url": {"url": b64_image}
            })
        else:
            print(f"Skipping image {image_url} for GPT-4o Vision due to loading error.")

    if not images_to_send_to_gpt:
        return json.dumps({
            "selected": [],
            "explanation": "Sağlanan görsellerden hiçbiri yüklenemedi veya işlenemediği için değerlendirme yapılamadı."
        })

    system_msg = {
        "role": "system",
        "content": """You are an expert hotel room selection assistant using image analysis.
        Your task is to carefully analyze the provided hotel room images based on the user's detailed request.
        For each image, identify relevant features and potential drawbacks.
        Finally, select the top 1-3 most suitable rooms and provide clear, concise justifications for each selection,
        as well as an overall explanation for your choices. Prioritize matching the user's specific desires."""
    }

    user_prompt_content_list = [
        {
            "type": "text",
            "text": f"""
I'm looking for a hotel room based on the following preferences:
📝 **My request:** "{user_prompt}"

You are evaluating {len(images_to_send_to_gpt)} hotel room images.
Please consider the following for each image (refer to them by their internal order, e.g., Image 1, Image 2, etc.):
- What are its key features (e.g., room type, bed type, view, amenities like balcony, air conditioning, fridge, desk)?
- How well does it match my request? Are there any aspects that don't fit perfectly?

After evaluating all images, select **up to 3 rooms** that are the best fit for my request. If fewer than 3 are highly relevant, select fewer.
Provide a **reason** for each selected room, explaining why it's a good match.
Also, give an **overall explanation** for your selection strategy.

✅ Respond STRICTLY in this JSON format:
{{
  "selected": [
    {{
      "index": 0,
      "reason": "This room clearly shows a comfortable double bed, a fridge, and a calming forest view, directly matching the user's primary criteria."
    }},
    {{
      "index": 2,
      "reason": "Image 3 features a large balcony that hints at a forest view, and a small fridge is visible, offering good comfort."
    }}
  ],
  "explanation": "The selected rooms were chosen because they visibly demonstrate a combination of a fridge and a forest view, prioritizing the user's core needs for comfort and specific amenities."
}}

If no suitable rooms are found among the provided images, return:
{{
  "selected": [],
  "explanation": ""Among the images on www.obilethotelfinderAI.com, no suitable room has yet been found that directly meets the user's criteria of 'comfortable room with refrigerator and forest view'. Some of them were observed to lack a refrigerator or have a view mismatch."
}}
"""
        }
    ]
    user_prompt_content_list.extend(images_to_send_to_gpt)

    user_msg = {
        "role": "user",
        "content": user_prompt_content_list
    }

    try:
        response = azure_client.chat.completions.create(
            model="gpt-4o",
            messages=[system_msg, user_msg],
            temperature=0.2,
            max_tokens=1000
        )
        result_content = response.choices[0].message.content
        try:
            parsed_json = json.loads(result_content)
        except json.JSONDecodeError:
            json_start = result_content.find('{')
            json_end = result_content.rfind('}') + 1
            if json_start != -1 and json_end != -1 and json_end > json_start:
                try:
                    parsed_json = json.loads(result_content[json_start:json_end])
                except json.JSONDecodeError:
                    print(f"WARNING: Could not extract valid JSON from response: {result_content}")
                    return json.dumps({"selected": [], "explanation": "GPT yanıtı beklendiği gibi JSON formatında değil ve otomatik düzeltilemedi."})
            else:
                print(f"WARNING: No JSON structure found in GPT response: {result_content}")
                return json.dumps({"selected": [], "explanation": "GPT yanıtı geçerli JSON içermiyor."})

        return json.dumps(parsed_json)
    except Exception as e:
        print(f"ERROR: [GPT4o Filter] GPT request error: {e}")
        return json.dumps({
            "selected": [],
            "explanation": f"GPT yanıtı alınamadı veya işlenirken genel bir hata oluştu: {e}"
        })


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
You are an expert hotel consultant and room image analyst for www.obilet.com, an online booking website. You will extract all the potential details for the customers' requests and queries.
You will observe much more on all the criteria required, especially for the following questions:
-Double rooms with a sea view,
-Rooms with a balcony and air conditioning, with a city view,
-Triple rooms with a desk,
-Rooms with a maximum capacity of 4 people,

For a given hotel room image, do the following:
1. Output a valid JSON object with these fields:
   {
     "room_type": "string",
     "view": "string",
     "bed_type": "string",
     "balcony": boolean,
     "jacuzzi": boolean,
     "air_conditioning": boolean,
     "desk": boolean,
     "max_capacity": integer
   }
2. Then, write a short, user-friendly full detailed description (1-2 sentences) of the room for a booking website.
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
                {"type": "text", "text": "Analyze this hotel room image and output the required JSON and caption."},
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
    Falls back to a static response if API fails.
    """
    try:
        # 1. Şehri AccuWeather Location API ile bul
        api_key = "PCfKgk3ntptMbI581tlvyc57gd9UzMsJ"  # .env'den oku veya config dosyandan çek
        location_url = f"http://dataservice.accuweather.com/locations/v1/cities/search?apikey={api_key}&q={city}&language=tr-tr"
        loc_resp = requests.get(location_url, timeout=5)
        loc_resp.raise_for_status()
        locations = loc_resp.json()
        if not locations:
            return f"{city} için hava durumu bulunamadı."
        location_key = locations[0]["Key"]

        # 2. Hava durumu bilgisini çek
        weather_url = f"http://dataservice.accuweather.com/currentconditions/v1/{location_key}?apikey={api_key}&language=tr-tr&details=true"
        weather_resp = requests.get(weather_url, timeout=5)
        weather_resp.raise_for_status()
        weather = weather_resp.json()[0]
        text = weather["WeatherText"]
        temp = weather["Temperature"]["Metric"]["Value"]
        feels = weather["RealFeelTemperature"]["Metric"]["Value"]
        return f"{city} için güncel hava durumu: {text}, sıcaklık: {temp}°C, hissedilen: {feels}°C."
    except Exception as e:
        print(f"[Weather API Error] {e}")
        return f"{city} için hava durumu şu anda alınamıyor."