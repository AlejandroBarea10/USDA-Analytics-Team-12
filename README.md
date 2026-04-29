# USDA Digital Intelligence Platform

## Descripción

Plataforma de análisis digital para USDA Rural Development que proporciona inteligencia sobre experiencia de usuarios, segmentación automática de páginas web y recomendaciones de optimización basadas en datos.

### Características Principales

- 📊 **Análisis de Analytics GA4**: Carga y procesa datos de Google Analytics 4 de forma automática
- 🔍 **Segmentación Inteligente**: Clustering K-Means para identificar patrones de experiencia de usuario
- 🎯 **Scoring Automático**: Engagement Score y Friction Index para priorizar mejoras
- 🤖 **Agente AI Estratégico**: Responde preguntas sobre los hallazgos y recomienda acciones
- 📈 **Visualizaciones Interactivas**: Gráficos Plotly para exploración de datos

## Instalación Local

### Requisitos
- Python 3.9+
- pip

### Pasos

1. **Clonar el repositorio**
   ```bash
   git clone <tu-repo>
   cd "Streamlit Final App"
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv .venv
   
   # En Windows:
   .venv\Scripts\activate
   
   # En macOS/Linux:
   source .venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ejecutar la aplicación**
   ```bash
   streamlit run "Final app.py"
   ```

5. **Acceder a la aplicación**
   - Abre tu navegador en: `http://localhost:8501`

## Estructura del Proyecto

```
.
├── Final app.py              # Aplicación principal Streamlit
├── DATA USDA.csv             # Datos de ejemplo (GA4)
├── requirements.txt          # Dependencias Python
├── .gitignore               # Archivos a ignorar en Git
└── README.md                # Este archivo
```

## Uso

1. **Cargar datos**: Sube un archivo CSV de Google Analytics 4 o usa el archivo de ejemplo
2. **Explorar resultados**: Visualiza métricas, clusters y análisis
3. **Consultar el agente**: Haz preguntas sobre los hallazgos y obtén recomendaciones

## Deploy en la Nube

### Opción 1: Streamlit Cloud (Recomendado)

1. Sube tu repositorio a GitHub
2. Ve a [streamlit.io/cloud](https://streamlit.io/cloud)
3. Conecta tu repositorio
4. Selecciona este proyecto
5. ¡Listo! Tu app estará disponible en línea

### Opción 2: Heroku

1. Crea un archivo `Procfile`:
   ```
   web: streamlit run "Final app.py"
   ```

2. Deploy:
   ```bash
   heroku create <app-name>
   git push heroku main
   ```

### Opción 3: Render

1. Conecta tu repositorio en [render.com](https://render.com)
2. Crea un nuevo Web Service
3. Build: `pip install -r requirements.txt`
4. Start: `streamlit run "Final app.py" --logger.level=error --server.enableXsrfProtection=false`

## Dependencias

- **streamlit**: Framework web para aplicaciones de datos
- **pandas**: Manipulación y análisis de datos
- **numpy**: Computación numérica
- **plotly**: Visualizaciones interactivas
- **scikit-learn**: Machine Learning (clustering, scaling, PCA)

## Autor

Purdue University - AI for Business Analytics

## Licencia

MIT
