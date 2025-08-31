import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

openai_endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
openai_key = os.getenv('AZURE_OPENAI_KEY')
accuwather_key = os.getenv('ACCUWEATHER_API_KEY')
weather_api_key = os.getenv('OPENWEATHER_API_KEY')  # OpenWeatherMap API key


