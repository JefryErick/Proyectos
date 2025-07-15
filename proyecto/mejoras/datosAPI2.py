import tweepy
import pandas as pd
from datetime import datetime
import time

# Configura tus credenciales (API v2)
client = tweepy.Client(
    bearer_token="AAAAAAAAAAAAAAAAAAAAAKbE2wEAAAAAEWmdcWY%2FDL%2BlfUf36rxc0RLRvEs%3DOwd54CXrBOgV5nai3029CYxsjMiaTkK9TIC09gBmDVwk91TG5R",  
    consumer_key="CHI05qX9UrwrPBYTaI1rRYiYx",
    consumer_secret="NL3KeoPpZR2hNfQUgRjW6g4lNhM1kJkiWbT0hy8jPOj6mvWWOc",
    access_token="1684675995477938176-A6ekHkQPGDxiimh2j4nDdwy6hrJFBB",
    access_token_secret="x8pXTKhVixfNOry6ESYIA5BANjVfmObUKliiJ7lrKbixu",
    wait_on_rate_limit=True
)

# Función para extraer retweets con paginación limitada (para pruebas)
def get_retweets_v2(tweet_id, pages_to_fetch=1):
    retweets = []
    try:
        print(f"🔍 Iniciando extracción de retweets del tweet {tweet_id}")
        for i, response in enumerate(tweepy.Paginator(
            client.get_retweeters,
            id=tweet_id,
            max_results=50,  # Máximo por página
            limit=pages_to_fetch  # Solo unas páginas para pruebas
        )):
            print(f"📦 Página {i+1}: {len(response.data) if response.data else 0} usuarios")
            time.sleep(1)
            if response.data:
                for user in response.data:
                    retweets.append([
                        user.username,  # user_who_retweeted
                        "original_user_placeholder",  # Lo actualizaremos después
                        datetime.now()  # timestamp
                    ])
        print(f"✅ Total de retweets extraídos: {len(retweets)}")
        return retweets
    except Exception as e:
        print(f"❌ Error durante extracción: {e}")
        return []

# ID del tweet (puedes cambiarlo por uno más pequeño para pruebas)
tweet_id = "1941289398257189291"

# Extraer retweets (solo 1 página de prueba)
retweets_data = get_retweets_v2(tweet_id, pages_to_fetch=1)

# Obtener el usuario original (una sola llamada)
if retweets_data:
    try:
        tweet_info = client.get_tweet(id=tweet_id, expansions=["author_id"], user_fields=["username"])
        original_user = tweet_info.includes["users"][0].username
        print(f"👤 Usuario original del tweet: @{original_user}")

        # Actualizar el campo original_user en los retweets
        for rt in retweets_data:
            rt[1] = original_user
    except Exception as e:
        print(f"❌ Error al obtener el usuario original: {e}")

# Guardar en CSV
if retweets_data:
    df = pd.DataFrame(retweets_data, columns=["user_who_retweeted", "original_user", "timestamp"])
    df.to_csv("retweets_dataset_v2.csv", index=False)
    print(f"📁 Dataset guardado exitosamente con {len(df)} retweets.")
else:
    print("⚠️ No se encontraron retweets o hubo un error.")
