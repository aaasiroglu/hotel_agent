import chainlit as cl
import json
from typing import Optional, List
from openai import AzureOpenAI
import config
import utils


@cl.step(type="tool")
async def search_hotel_rooms(query: str) -> str:
    return utils.vector_search_hotel_rooms(query,5)


@cl.step(type="tool")
async def get_weather(city: str) -> str:
    return utils.get_current_weather(city)

tools = [
            {
            "type": "function",
            "function": {
                "name": "search_hotel_rooms",
                "description": "Search for hotel rooms based on the user's query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The query to search for hotel rooms."
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
                "description": "Get the current weather for a specified city.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "The name of the city to get the weather for."
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
                elif function_name == "search_hotel_rooms":
                    tool_result = await search_hotel_rooms(function_args["query"])
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
    # weather_tool = WeatherTool()
    # cl.user_session.set("weather_tool", weather_tool)
    cl.user_session.set("messages", [
        {"role": "system", "content": '''You are a helpful hotel finder assistant. You work for a hotel booking service "www.obilet.com" and your task is to help customers find the best hotel rooms for their needs.
         You can answer questions about the weather, hotel rooms, and general inquiries.
         Use tools when necessary.
         
         While showing hotels results, list rooms with following format:
            **Room Name**: Best Room For You
            - **Description**: A spacious room with a king-size bed and a beautiful view. You can enjoy the sea breeze from your private balcony.
            - **Key Attributes**: Air Conditioning, Sea View, King Bed, Fridge
            - **Image**: ![Room Image](http://static.obilet....1.jpg)

        IMPORTANT: Do not use any other format, just use the above format. Image URLs should start with only "http", if they start with "https".
         '''},
    ])

@cl.on_message
async def on_message(message: cl.Message):
    messages = cl.user_session.get("messages")
    messages.append({"role": "user", "content": message.content})
    
    content = await call_azure_openai(messages)
    #convert extract image urls to markdown
    
    answer = cl.Message(content=content, author="Assistant")
    
    messages.append({"role": "assistant", "content": content})
    cl.user_session.set("messages", messages)
    await answer.send()

@cl.set_starters
async def set_starts() -> List[cl.Starter]:
    return [
        cl.Starter(
            label="Double rooms with a sea view",
            message="Double rooms with a sea view",
        ),
        cl.Starter(
            label="Rooms with a balcony and air conditioning, with a city view",
            message="Rooms with a balcony and air conditioning, with a city view",
        ),
        cl.Starter(
            label="Triple rooms with a desk",
            message="Triple rooms with a desk",
        ),
        cl.Starter(
            label="Rooms with a maximum capacity of 4 people",
            message="Rooms with a maximum capacity of 4 people",
        )
    ]
