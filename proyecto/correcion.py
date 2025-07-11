import pandas as pd
import numpy as np
from datetime import timedelta
import random

def variar_timestamp(timestamp):
    # Aplicar diferentes niveles de variación
    variacion_dias = random.randint(0, 32)  # Aumenté el rango para permitir cambios de mes
    variacion_horas = random.randint(0, 23)  # Hasta 23 horas
    variacion_minutos = random.randint(0, 59)  # Hasta 59 minutos
    variacion_segundos = random.randint(0, 59)  # Hasta 59 segundos
    variacion_microsegundos = random.randint(0, 999999)  # Hasta 999999 microsegundos
    
    # Aplicar las variaciones (algunas pueden ser negativas para variar en ambos sentidos)
    nuevo_timestamp = timestamp + timedelta(
        days=variacion_dias * random.choice([-1, 1]),
        hours=variacion_horas * random.choice([-1, 1]),
        minutes=variacion_minutos * random.choice([-1, 1]),
        seconds=variacion_segundos,
        microseconds=variacion_microsegundos
    )
    
    return nuevo_timestamp  # Eliminamos la verificación del mes

def ajustar_timestamps(input_csv, output_csv):
    # Leer el archivo CSV
    df = pd.read_csv(input_csv)
    
    # Convertir la columna timestamp a datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Aplicar la variación a cada timestamp
    df['timestamp'] = df['timestamp'].apply(variar_timestamp)
    
    # Guardar el nuevo CSV
    df.to_csv(output_csv, index=False)
    print(f"Archivo guardado en: {output_csv}")

# Uso del programa
input_file = 'twitter_retweets_ajustado.csv'
output_file = 'twitter_retweets_ajustado2.csv'

ajustar_timestamps(input_file, output_file)