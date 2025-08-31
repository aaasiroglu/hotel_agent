import chainlit as cl
import json
from typing import Optional, List
from openai import AzureOpenAI
import config
import utils


@cl.step(type="tool")
async def search_shoes(query: str) -> str:
    return utils.vector_search_shoes(query, 5)


@cl.step(type="tool")
async def get_weather(city: str) -> str:
    return utils.get_current_weather(city)


@cl.step(type="tool")
async def plan_and_search(user_request: str) -> str:
    # 1) Intent analizi - akıcı ve hassas
    plan = await utils.extract_shoe_intent(user_request)
    
    # Premium analiz gösterimi
    intent_details = f"""✨ **Kişisel Analiz Tamamlandı**

🎯 **Size Özel Arama Kriterleri:**
- 👟 **Aradığınız Tip:** {plan.get('shoe_type', 'Açık tercih belirtilmedi')}
- 🎨 **Renk Tercihi:** {plan.get('color', 'Renk tercihi esnek')}
- 💫 **Stil Yaklaşımı:** {plan.get('style', 'Stil tercihi geniş')}
- � **Kullanım Amacı:** {plan.get('use_case', 'Genel kullanım')}"""

    if plan.get('special_features'):
        intent_details += f"\n- ✅ **Özel İstekleriniz:** {', '.join(plan['special_features'])}"
        
    intent_details += f"\n\n� **Arama Sorgusu:** `{plan.get('query', user_request)}`"
    intent_details += "\n\n🎁 **Size özel seçilmiş ayakkabılar yükleniyor...**\n\n"
    
    # 2) Optimize edilmiş arama
    results = await search_shoes(plan["query"])
    
    # 3) Premium sonuç formatı
    if "bulunamadı" in results or "bulunmuyor" in results:
        return intent_details + f"💔 **Üzgünüm!** Tam bu kriterlere uygun ayakkabı bulamadım.\n\n🤔 **Alternatif Öneriler:** Lütfen:\n- Renk tercihlerinizi genişletin\n- Farklı stillerle ilgili açık olun\n- Benzer özelliklerdeki ayakkabıları değerlendirin\n\n{results}"
    
    return intent_details + results + "\n\n💎 **Bu seçenekler arasından hangisi daha çok ilginizi çekiyor? Daha detaylı bilgi veya alternatifler istiyorsanız sormaktan çekinmeyin!**"


tools = [
            {
            "type": "function",
            "function": {
                "name": "plan_and_search",
                "description": "Detaylı kullanıcı taleplerini analiz eder ve ayakkabı veritabanında aramak için yapılandırılmış bir sorgu oluşturur. Stil, amaç, renk gibi birden çok kriter içeren talepler için kullanılır.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "user_request": {
                            "type": "string",
                            "description": "Kullanıcının aradığı ayakkabıyı tanımlayan doğal dil sorgusu. Örnek: 'düğün için şık topuklu ayakkabı'."
                        }
                    },
                    "required": ["user_request"]
                }
            }
        },
            {
            "type": "function",
            "function": {
                "name": "search_shoes",
                "description": "Veritabanında doğrudan ayakkabı araması yapar. 'siyah bot' gibi net ve basit sorgular için kullanılır.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Veritabanında aranacak net sorgu metni."
                        }
                    },
                    "required": ["query"]
                }
            }
        },
            {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Belirtilen bir şehir için güncel hava durumu bilgisini alır. Mevsimlik ayakkabı önerileri için kullanılabilir.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "Hava durumu bilgisi alınacak şehrin adı."
                        }
                    },
                    "required": ["city"]
                }
            }
        }
    ]

# Initialize Azure OpenAI client
client = AzureOpenAI(
    api_key=config.openai_key,
    api_version="2024-02-15-preview",
    azure_endpoint=config.openai_endpoint,
    default_headers={"Ocp-Apim-Subscription-Key": config.openai_key}
)

async def call_azure_openai(messages):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",  # deployment name
            messages=messages,
            stream=False,
            temperature=0.7,
            max_tokens=800,
            tools=tools,
        )
        message = response.choices[0].message

        # Check if the message contains tool calls
        if message.tool_calls:
                    
            # Process each tool call
            for tool_call in message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                
                # Execute the appropriate tool
                if function_name == "get_weather":
                    tool_result = await get_weather(function_args["city"])
                elif function_name == "search_shoes":
                    tool_result = await search_shoes(function_args["query"])
                elif function_name == "plan_and_search":
                    tool_result = await plan_and_search(function_args["user_request"])
                else:
                    tool_result = f"Tool {function_name} not found"
                
                # Append the tool result to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(tool_result)
                })
            
            # Get the final response with tool results
            final_response = client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                stream=False,
                temperature=0.7,
                max_tokens=800,
                tools=tools,
            )
            return final_response.choices[0].message.content
        
        return message.content

    except Exception as e:
        print(f"Error calling Azure OpenAI: {str(e)}")
        return "I apologize, but I encountered an error. Please try again."

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("messages", [
        {"role": "system", "content": '''Sen Premium Shoes AI'ın kişisel ayakkabı danışmanısın. 🎯

� KİŞİLİK:
- Sıcak, samimi ve profesyonel
- Moda ve stil konularında uzman
- Her müşterinin benzersiz ihtiyaçlarını anlayan
- Hevesli ama yorum dayatmayan

🎯 GÖREVİN:
1. MÜŞTERIYI ANLA: "Hangi etkinlik için?", "Tarzınız nedir?", "Hangi renkler seversiniz?"
2. AKILLI ARAMA: Karmaşık istekler için plan_and_search, basit aramalar için search_shoes kullan
3. MÜKEMMEL SUNUM: Bulunan ürünleri görsellerle ve detaylarla sun ama asla veritabanında olmayan bir modeli uydurma
4. SATIŞI TAMAMLA: Seçim yapmasına yardım et, öneriler sun

� ARAMA STRATEJISİ:
- Duygusal bağlantı kur: "Düğününüz için mükemmel olacak!"
- Detaylı özellikler belirt: Rahat mı? Su geçirmez mi? 
- Stil önerileri ver: "Bu renk abiyenizle muhteşem uyum sağlayacak"

⚠️ DİKKAT: SADECE veritabanındaki gerçek ürünleri göster. 

�️ KONUŞMA TARZI: Dostane, profesyonel ve yardımsever. İlgili bir satış danışmanı gibi "Size özel seçtim..." gibi kişisel dokunuşlar ekle.'''},
    ])
    
    

@cl.on_message
async def on_message(message: cl.Message):
    thinking_messages = [
        "Sizin için en uygun ayakkabıları arıyorum...",
        "Tercihleriniz analiz ediliyor...",
        "En iyi eşleşmeler bulunuyor...",
        "Öneriler hazırlanıyor..."
    ]
    
    import random
    thinking_msg = random.choice(thinking_messages)
    thinking = cl.Message(content=thinking_msg, author="Ayakkabı Asistanı 👟")
    await thinking.send()
    
    messages = cl.user_session.get("messages")
    messages.append({"role": "user", "content": message.content})
    
    try:
        content = await call_azure_openai(messages)
        
        if "bulundu" in content:
            content += "\n\n💎 **Kişisel Önerim:** Bu seçenekler arasından hangisi size daha yakın geliyor? Daha fazla detay veya alternatif istiyorsanız sormaktan çekinmeyin!"
        elif "bulunamadı" in content or "bulunmuyor" in content:
            content += "\n\n🤔 **Alternatif Çözüm:** İstediklerinize benzer farklı seçenekleri araştırabilirim. Tercihlerinizi biraz daha detaylandırabilir misiniz?"
            
    except Exception as e:
        content = "😔 Üzgünüm, şu anda teknik bir sorun yaşıyoruz. Lütfen birkaç saniye sonra tekrar deneyin veya farklı kelimelerle arama yapın."
        print(f"Error: {e}")
    
    await thinking.remove()
    
    answer = cl.Message(content=content, author="Ayakkabı Asistanı 👟")
    
    messages.append({"role": "assistant", "content": content})
    cl.user_session.set("messages", messages)
    await answer.send()

@cl.set_starters
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="💃 Gala Etkinliği İçin Şık Topuklu",
            message="Bu akşam bir gala etkinliğine katılacağım. Bordo bir elbise giyeceğim ve çok şık görünmek istiyorum. Hangi topuklu ayakkabı önerirsiniz?",
        ),
        cl.Starter(
            label="👔 Yeni İş İçin Formal Ayakkabı",
            message="Önümüzdeki hafta yeni işime başlıyorum. Hem şık hem de tüm gün rahat olacağım formal bir ayakkabı önerebilir misiniz?",
        ),
        cl.Starter(
            label="❄️ Kış İçin Sıcak ve Stylish Bot",
            message="Kış geliyor ve hem şık görünmek hem de ayaklarımı sıcak tutmak istiyorum. Su geçirmez özellikli şık bot önerileriniz var mı?",
        ),
        cl.Starter(
            label="🏃‍♀️ Spor ve Günlük Kombini",
            message="Hem spor yaparken hem de günlük hayatta giyebileceğim, tarzımı yansıtan ve ultra rahat bir ayakkabı arıyorum.",
        ),
        cl.Starter(
            label="🌟 Trend ve Benzersiz Bir Şeyler",
            message="Sıradanlıktan sıkıldım! 2025'in trend ayakkabı modellerinden, göz alıcı ve benzersiz bir şeyler önerir misiniz?",
        )
    ]
