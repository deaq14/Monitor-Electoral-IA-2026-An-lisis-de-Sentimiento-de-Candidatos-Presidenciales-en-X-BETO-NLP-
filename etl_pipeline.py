import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline
import logging
import random
from datetime import datetime, timedelta
import numpy as np

# ==========================================
# 1. CONFIGURACIÓN DE LOGS
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.FileHandler("proyecto_sentimiento.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ==========================================
# 2. IMPORTACIÓN SEGURA DE SNSCRAPE
# ==========================================
# Python 3.12+ rompe snscrape. Esto evita que el programa colapse.
try:
    import snscrape.modules.twitter as sntwitter
    SNSCRAPE_DISPONIBLE = True
except (ImportError, AttributeError) as e:
    logger.warning(f"⚠️ No se pudo importar snscrape (Incompatible con tu Python): {e}")
    logger.warning("➡️ El sistema funcionará EXCLUSIVAMENTE con datos simulados.")
    SNSCRAPE_DISPONIBLE = False

# Configuración Global
START_DATE = "2025-11-15"
END_DATE = "2025-12-05"
CANDIDATOS = ["Abelardo de la Espriella", "Iván Cepeda", "Sergio Fajardo"]

# ==========================================
# 3. FUNCIONES AUXILIARES
# ==========================================

def _descargar_recursos_nltk():
    """Descarga recursos de NLTK de forma segura."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        logger.info("Descargando recurso NLTK: punkt")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        logger.info("Descargando recurso NLTK: stopwords")
        nltk.download('stopwords', quiet=True)

def _generar_datos_simulados(candidatos, n=200):
    """Fallback: Genera datos falsos si el scraping falla."""
    logger.info("🎲 Generando datos simulados para demostración...")
    data = []
    textos_ejemplo = [
        "Es el mejor candidato para el país, tiene mi voto.",
        "No me convence su propuesta, es un desastre total.",
        "El debate de ayer estuvo interesante, aunque le faltó fuerza.",
        "Totalmente en contra de sus políticas corruptas.",
        "Vamos con toda por el cambio que necesitamos.",
        "Es más de lo mismo, no le creo nada.",
    ]
    
    start = datetime.strptime(START_DATE, "%Y-%m-%d")
    end = datetime.strptime(END_DATE, "%Y-%m-%d")
    
    for _ in range(n):
        cand = random.choice(candidatos)
        delta = end - start
        random_days = random.randrange(delta.days)
        date = start + timedelta(days=random_days)
        
        # Añadimos ruido aleatorio al texto
        txt = random.choice(textos_ejemplo) + f" ({cand})"
        
        data.append([
            date,
            txt,
            random.randint(100, 50000), # Followers
            cand
        ])
    
    return pd.DataFrame(data, columns=['Date', 'Text', 'Followers', 'Candidate'])

# ==========================================
# 4. LÓGICA PRINCIPAL (ETL)
# ==========================================

def obtener_datos(usar_api_real=False):
    """
    Intenta obtener datos reales. Si falla o la librería no existe, usa simulados.
    """
    # Si snscrape falló al importar, forzamos simulados
    if not SNSCRAPE_DISPONIBLE:
        logger.info("🔒 Modo Forzado: Simulados (Librería de scraping no disponible).")
        return _generar_datos_simulados(CANDIDATOS)

    try:
        if not usar_api_real:
            logger.info("Modo Scraping Real desactivado por configuración. Usando simulados.")
            return _generar_datos_simulados(CANDIDATOS)

        logger.info("Iniciando Scraping de X (Twitter)...")
        tweets_list = []
        for candidato in CANDIDATOS:
            query = f'"{candidato}" since:{START_DATE} until:{END_DATE} lang:es'
            try:
                # Intentamos obtener 50 tweets por candidato
                gen = sntwitter.TwitterSearchScraper(query).get_items()
                for i, tweet in enumerate(gen):
                    if i >= 50: break
                    tweets_list.append([
                        tweet.date,
                        tweet.rawContent,
                        tweet.user.followersCount,
                        candidato
                    ])
                logger.info(f"✅ Scraping exitoso para: {candidato}")
            except Exception as e:
                logger.error(f"❌ Error al scrapear candidato {candidato}: {e}")
                continue 
        
        if not tweets_list:
            logger.warning("No se encontraron tweets reales. Cambiando a simulados.")
            return _generar_datos_simulados(CANDIDATOS)
            
        return pd.DataFrame(tweets_list, columns=['Date', 'Text', 'Followers', 'Candidate'])

    except Exception as e:
        logger.critical(f"🔥 Fallo crítico en recolección de datos: {e}")
        return _generar_datos_simulados(CANDIDATOS)

def procesar_texto_y_sentimiento(df):
    """
    Toma el DF crudo, limpia texto con NLTK y clasifica con BETO.
    """
    _descargar_recursos_nltk()
    stop_words = set(stopwords.words('spanish'))

    # 1. Cargar Modelo
    try:
        logger.info("Cargando modelo BETO (pysentimiento/robertuito)...")
        # NOTA: La primera vez descargará aprox 500MB
        classifier = pipeline("sentiment-analysis", model="pysentimiento/robertuito-sentiment-analysis")
    except Exception as e:
        logger.error(f"Error cargando el modelo de IA: {e}")
        return pd.DataFrame() 

    # 2. Funciones Internas
    def limpiar(text):
        try:
            tokens = word_tokenize(text.lower(), language='spanish')
            return " ".join([w for w in tokens if w.isalnum() and w not in stop_words])
        except Exception:
            return text

    def clasificar(text):
        try:
            # Truncamos a 512 caracteres
            pred = classifier(text[:512])[0]
            label = pred['label']
            
            val = 1 if label == 'POS' else (-1 if label == 'NEG' else 0)
            return label, val
        except Exception as e:
            return 'NEU', 0

    logger.info("Procesando textos y calculando sentimientos...")
    
    df['Clean_Text'] = df['Text'].apply(limpiar)
    
    # Aplicar modelo
    sentimientos = df['Clean_Text'].apply(lambda x: clasificar(x))
    
    df['Sentiment_Label'] = [x[0] for x in sentimientos]
    df['Sentiment_Value'] = [x[1] for x in sentimientos]
    
    # Calcular Influencia 
    df['Influence_Score'] = np.log1p(df['Followers']) * df['Sentiment_Value']
    
    # Convertir fecha para agrupar (manejo de formato simulado vs real)
    df['Date_Only'] = pd.to_datetime(df['Date']).dt.date
    
    logger.info(f"Procesamiento completado. {len(df)} registros listos.")
    return df

def ejecutar_pipeline():
    # Nota: Pasamos usar_api_real=False por defecto para evitar errores de conexión ahora mismo
    df_raw = obtener_datos(usar_api_real=False) 
    df_clean = procesar_texto_y_sentimiento(df_raw)
    return df_clean