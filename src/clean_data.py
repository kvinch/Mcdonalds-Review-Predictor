import pandas as pd
import re
import os


def convert_time(text):
    if not isinstance(text, str):
        return None 
    text = text.lower()
    
    if "a day" in text or "an hour" in text or "a week" in text or "a month" in text or "a year" in text:
        num = 1
    else:
        
        match = re.search(r"\d+", text)
        if match is None:
            return None
        num = int(match.group())
    
    if "hour" in text or "minute" in text or "second" in text:
        return 0 
    elif "day" in text:
        return num
    elif "week" in text:
        return num * 7
    elif "month" in text:
        return num * 30
    elif "year" in text:
        return num * 365
    else:
        return None

def clean_data(input_path, output_path):
    print(f"Reading raw data from {input_path}...")
    # Cargas el data set mediante el encoding latin1 para evitar errores de codificación
    df = pd.read_csv(input_path, encoding="latin1")
    df["review"] = df["review"].str.replace("ï¿½", "", regex=False)
    # 1. Se eliminan los espacios en blanco de las columnas
    df.columns = df.columns.str.strip()
    print("Column names cleaned.")
    
    # 2. Se eliminan las columnas irrelevantes
    columns_to_drop = ["reviewer_id", "store_name", "category"]
    df = df.drop(columns=columns_to_drop, errors="ignore")
    print(f"Dropped columns: {columns_to_drop}")
    
    # 3. Se eliminan las filas con valores nulos en las columnas latitude y longitude 
    initial_shape = df.shape
    df = df.dropna(subset=["latitude", "longitude"])
    print(f"Dropped {initial_shape[0] - df.shape[0]} rows due to nulls in latitude/longitude.")
    
    # 4. Se homogeneizan los tipos de datos y los formatos
    #  Se normalizan los datos para evitar incongruencias, ej: 1,240 -> 1240
    if "rating_count" in df.columns:
        df["rating_count"] = df["rating_count"].astype(str).str.replace(",", "").astype(int)
    
    # Se eliminan los caracteres no numéricos de la columna rating y se convierte en entero
    if "rating" in df.columns:
        df["rating"] = df["rating"].astype(str).str.extract(r"(\d+)").astype(int)
        
    # Realizamos la transformación de los datos de review_time como dias para que sean más homogéneos y los limpiamos
    if "review_time" in df.columns:
        df["review_time_since_days"] = df["review_time"].apply(convert_time)
        df.drop(columns=["review_time"], inplace=True)

    # Se normalizan los valores de la columna review para que no contengan saltos de línea
    if "review" in df.columns:
        df["review"] = df["review"].astype(str).str.replace(r"\r", "", regex=True)
        df["review"] = df["review"].astype(str).str.replace(r"\n", " ", regex=True)
        df["review"] = df["review"].str.strip()
        
        
    # Clean 'store_address': Trim whitespace
    if "store_address" in df.columns:
        df["store_address"] = df["store_address"].astype(str).str.strip()
    
    # 5. Se eliminan las filas duplicadas
    initial_shape = df.shape
    df = df.drop_duplicates(keep="first")
    print(f"Dropped {initial_shape[0] - df.shape[0]} duplicate rows.")
    
    # Se resetea el index
    df = df.reset_index(drop=True)
    
    # 6. Se guarda el dataset limpio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Clean data saved to {output_path} with shape {df.shape}")

if __name__ == "__main__":
    # Se definen las rutas relativas al archivo actual
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    input_file = os.path.join(BASE_DIR, "data", "McDonalds_RAW.csv")
    output_file = os.path.join(BASE_DIR, "data", "McDonalds_Clean.csv")
    
    if os.path.exists(input_file):
        clean_data(input_file, output_file)
    else:
        print(f"Error: Input file not found at {input_file}")
