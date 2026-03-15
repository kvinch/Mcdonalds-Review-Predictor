# McDonald's Review Predictor
## Análisis de Datos y Machine Learning sobre Reseñas de McDonald's
Un proyecto completo de **Data Science** que abarca desde la limpieza de datos crudos hasta el despliegue de un modelo de Machine Learning en un dashboard interactivo. El objetivo es predecir si una reseña de McDonald's es **buena**, **neutral** o **mala** a partir de características textuales y de sentimiento.

---

## ¿Qué hace este proyecto?
Dado un dataset de reseñas reales de McDonald's, el proyecto:
1. **Limpia** los datos crudos eliminando ruido, nulos y columnas irrelevantes.
2. **Extrae features** mediante NLP (análisis de sentimiento con VADER y TextBlob) y feature engineering.
3. **Entrena** un modelo Random Forest para clasificar cada reseña como `good`, `neutral` o `bad`.
4. **Visualiza** el análisis exploratorio en un notebook interactivo.
5. **Despliega** un dashboard con Streamlit donde se puede explorar el dataset y hacer predicciones en tiempo real.

---

## Estructura del Proyecto
```
Mcdonalds-Review-Predictor/
│
├── data/
│   ├── McDonalds_RAW.csv 
│   ├── McDonalds_Clean.csv
│   └── processed/
│       └── McDonalds_Features.csv
│
├── src/
│   ├── clean_data.py 
│   ├── data_features+NLP.py 
│   └── ML-RandomForest.py 
│
├── model/
│   └── rf_model.pkl
│
├── notebooks/
│   └── EDA.ipynb 
│
├── dashboard/
│   └── dashboard.py 
│
├── requirements.txt 
└── README.md
```

## Modelo de Machine Learning
| Parámetro | Valor |
|---|---|
| Algoritmo | Random Forest Classifier |
| Estimadores | 300 árboles |
| Balance de clases | `class_weight='balanced'` |
| Accuracy TEST | **72.6%** |
| Cross-Val (5-fold) | **72.3% ± ~1%** |
| Target | `rating_label` → `good` / `neutral` / `bad` |

### Features utilizadas
| Feature | Descripción |
|---|---|
| `sentiment_compound` | Score VADER (−1 a +1) |
| `sentiment_polarity` | Polaridad TextBlob (−1 a +1) |
| `sentiment_subjectivity` | Subjetividad TextBlob (0 a 1) |
| `sentiment_label_vader` | Clasificación VADER (positive/neutral/negative) |
| `review_length` | Longitud del texto en caracteres |
| `review_word_count` | Número de palabras |
| `review_time_since_days` | Días desde que se publicó la reseña |
| `rating_count` | Total de ratings del local |
| `location_cluster` | Cluster geográfico del restaurante (0–9) |
| `city_enc` / `postal_code_enc` | Ciudad y código postal codificados |

## Dashboard (Streamlit)
El dashboard permite:
-  Ver las métricas de rendimiento del modelo
-  Leer los insights clave del análisis
-  Explorar los 3 datasets (RAW, Limpio, Preprocesado) en tabla interactiva
-  Hacer predicciones interactivas introduciendo valores manualmente

### Pasos de instalación
```bash
# 1. Instalar dependencias
pip install -r requirements.txt
# 2. Descargar recursos de NLTK (primera vez)
python -c "import nltk; nltk.download('vader_lexicon')"
# 3. Lanzar el dashboard
streamlit run dashboard/dashboard.py
```

## Enlace del Dataset de _Kaggle_
```bash
https://www.kaggle.com/datasets/nelgiriyewithana/mcdonalds-store-reviews
```
