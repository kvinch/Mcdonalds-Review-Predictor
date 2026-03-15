import pandas as pd
import numpy as np
import os
import re
import warnings
warnings.filterwarnings("ignore")

# Sentiment Analysis
from textblob import TextBlob
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)
try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)

from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Clustering
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# 1. Review rating
def create_rating_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea la columna 'rating_label':
      > 4  : 'good'
      < 3  : 'bad'
      == 3 : 'neutral'
    """
    def label(r):
        if r > 4:
            return "good"
        elif r < 3:
            return "bad"
        else:
            return "neutral"

    df["rating_label"] = df["rating"].apply(label)
    print(f"[✓] rating_label  : distribución:\n{df['rating_label'].value_counts().to_string()}\n")
    return df


# 2. Crear el Review Length -> Longitud de la reseña
def create_review_length(df: pd.DataFrame) -> pd.DataFrame:
    """Cuenta caracteres en la columna 'review'."""
    df["review_length"] = df["review"].astype(str).apply(len)
    print(f"[✓] review_length : min={df['review_length'].min()}, "
          f"max={df['review_length'].max()}, "
          f"media={df['review_length'].mean():.1f}\n")
    return df


# 3. Location cluster (lat + lon : KMeans)
def create_location_cluster(df: pd.DataFrame, n_clusters: int = 10) -> pd.DataFrame:
    """
    Combina latitude y longitude en un clúster espacial mediante KMeans.
    El número de clústeres se elige de forma adaptativa si el dataset
    tiene menos ubicaciones únicas que n_clusters.
    """
    coords = df[["latitude", "longitude"]].copy()
    unique_coords = coords.drop_duplicates().shape[0]
    k = min(n_clusters, unique_coords)

    scaler = StandardScaler()
    coords_scaled = scaler.fit_transform(coords)

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df["location_cluster"] = kmeans.fit_predict(coords_scaled)

    print(f"[✓] location_cluster : {k} clústeres creados\n")
    return df


# 4. Parseo de store_address : calle, ciudad, código postal
def parse_store_address(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formato esperado: "STREET, CITY, STATE ZIP, United States"
    """
    def extract_parts(address: str):
        address = str(address).strip()

        address = re.sub(r",?\s*United States\s*$", "", address, flags=re.IGNORECASE).strip()

        parts = [p.strip() for p in address.split(",")]

        street      = parts[0] if len(parts) > 0 else np.nan
        city        = np.nan
        postal_code = np.nan

        if len(parts) >= 2:
            city = parts[1].strip()

        # El código postal suele ir en el último segmento "STATE ZIP"
        if len(parts) >= 3:
            last_segment = parts[-1].strip()
            zip_match = re.search(r"\b(\d{5})\b", last_segment)
            if zip_match:
                postal_code = zip_match.group(1)

        return street, city, postal_code

    results = df["store_address"].apply(extract_parts)
    df["street"]      = [r[0] for r in results]
    df["city"]        = [r[1] for r in results]
    df["postal_code"] = [r[2] for r in results]

    return df


# 5. Sentiment analysis 
def create_sentiment_features(df: pd.DataFrame) -> pd.DataFrame:
    # VADER
    sid = SentimentIntensityAnalyzer()
    def vader_compound(text):
        try:
            return sid.polarity_scores(str(text))["compound"]
        except Exception:
            return 0.0

    def vader_label(score):
        if score >= 0.05:
            return "positive"
        elif score <= -0.05:
            return "negative"
        else:
            return "neutral"

    df["sentiment_compound"] = df["review"].apply(vader_compound)
    df["sentiment_label_vader"] = df["sentiment_compound"].apply(vader_label)

    # TextBlob
    def tb_polarity(text):
        try:
            return TextBlob(str(text)).sentiment.polarity
        except Exception:
            return 0.0

    def tb_subjectivity(text):
        try:
            return TextBlob(str(text)).sentiment.subjectivity
        except Exception:
            return 0.5

    df["sentiment_polarity"]     = df["review"].apply(tb_polarity)
    df["sentiment_subjectivity"] = df["review"].apply(tb_subjectivity)
    return df


# 6. Transformaciones adicionales
def additional_transformations(df: pd.DataFrame) -> pd.DataFrame:

    # a) Rating normalizado (1-5 : 0-1)
    df["rating_normalized"] = (df["rating"] - 1) / 4.0
    print("[✓] rating_normalized creado (escala 0-1)")

    # b) Conteo de palabras
    df["review_word_count"] = df["review"].astype(str).apply(
        lambda x: len(x.split())
    )
    print("[✓] review_word_count creado")

    # c) Diferencia entre rating y sentimiento (posible inconsistencia)
    df["rating_vs_sentiment_gap"] = (
        df["rating_normalized"] - df["sentiment_compound"]
    ).round(4)
    print("rating_vs_sentiment_gap creado\n")

    return df


# Main pipeline
def feature_engineering(input_path: str, output_path: str) -> pd.DataFrame:
    print(f"\n{'='*60}")
    print(f"  DATA FEATURE ENGINEERING - McDonalds Reviews")
    print(f"{'='*60}\n")

    # Carga del CSV limpio
    df = pd.read_csv(input_path, encoding="utf-8")
    print(f"Dataset cargado: {df.shape[0]:,} filas x {df.shape[1]} columnas\n")

    # Aplicar transformaciones en orden
    df = create_rating_label(df)
    df = create_review_length(df)
    df = create_location_cluster(df, n_clusters=10)
    df = parse_store_address(df)
    df = create_sentiment_features(df)
    df = additional_transformations(df)

    # Eliminamos las columnas que ya no tienen relevancia
    df = df.drop(["store_address", "latitude", "longitude", "review"], axis=1)

    # Guardar resultado
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="latin1")

    print(f"\n{'='*60}")
    print(f"\n  Columnas nuevas añadidas:")
    new_cols = [
        "rating_label", "review_length", "location_cluster",
        "street", "city", "postal_code",
        "sentiment_compound", "sentiment_label_vader",
        "sentiment_polarity", "sentiment_subjectivity",
        "rating_normalized", "review_word_count", "rating_vs_sentiment_gap"
    ]
    for col in new_cols:
        if col in df.columns:
            print(f"    • {col}")
    print(f"{'='*60}\n")
    return df


if __name__ == "__main__":
    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file  = os.path.join(BASE_DIR, "data", "McDonalds_Clean.csv")
    output_file = os.path.join(BASE_DIR, "data", "processed", "McDonalds_Features.csv")

    if os.path.exists(input_file):
        feature_engineering(input_file, output_file)
    else:
        print(f"No se pudo encontrar el archivo: {input_file}")
