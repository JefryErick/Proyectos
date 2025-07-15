# Proyecto de Estadística Computacional 2

Analizador de Propagación de Información con Grafos Temporales
============================================================

¿QUÉ BUSCA HACER ESTE CÓDIGO?
-----------------------------
Este proyecto implementa un Analizador de Propagación de Información con Grafos Temporales.
Su objetivo es rastrear, analizar y visualizar cómo se difunde la información en redes sociales, usando algoritmos de grafos y técnicas estadísticas modernas.

Características principales:
- Monitor de difusión en tiempo real: Permite cargar datasets de redes sociales y analizar cómo se propaga la información entre usuarios.
- Bootstrap temporal: Evalúa la robustez de los resultados mediante remuestreo (bootstrap) de las secuencias de propagación.
- Validación cruzada temporal: Evalúa la capacidad predictiva de los modelos de difusión usando técnicas de validación cruzada adaptadas a datos temporales.
- Métricas de viralidad: Calcula indicadores clave sobre la "fuerza" y alcance de la propagación.
- Visualización interactiva: Presenta los resultados en un dashboard moderno y fácil de interpretar.


DATASETS DISPONIBLES
--------------------
- Twitter Cascade Dataset: https://snap.stanford.edu/data/twitter-2010.html
- Digg Social News: https://snap.stanford.edu/data/digg-friends.html
- Memetracker: https://snap.stanford.edu/data/memetracker9.html
- Information Cascades: https://www.isi.edu/~lerman/downloads/digg2009.html


EXPLICACIÓN DE CADA CUADRO Y RESULTADO DEL DASHBOARD
-----------------------------------------------------

🔬 Resultados de Validación Cruzada Temporal
-------------------------------------------
Evalúa la capacidad de los modelos para predecir la propagación de información, usando "folds" temporales (se entrena en un periodo y se prueba en otro).
- MAE: Error absoluto medio (qué tan lejos están las predicciones de la realidad).
- RMSE: Error cuadrático medio (penaliza más los errores grandes).
- R²: Qué porcentaje de la variabilidad de los datos explica el modelo (1 es perfecto).
- Robustez: Qué tan estables son los resultados ante cambios en los datos.

📈 Bootstrap de Velocidades de Propagación
-----------------------------------------
Un histograma que muestra la distribución de velocidades a la que se propaga la información, calculada mediante remuestreo (bootstrap).
Permite ver si la difusión es rápida o lenta y cuánta variabilidad hay entre diferentes cascadas.

🎯 Distribución de Alcances
--------------------------
Un histograma que muestra cuántos usuarios (alcance) logra cada cascada de información.
Permite identificar si la mayoría de las cascadas son pequeñas o si hay algunas muy virales.

⚡ Métricas de Viralidad
-----------------------
Muestra indicadores clave de "fuerza viral" de la información, como el factor de viralización, profundidad promedio y ancho promedio de las cascadas.
Ayuda a entender si la información se propaga de forma superficial (pocos niveles) o profunda (muchos niveles).

🔄 Comparación de Modelos (CV)
-----------------------------
Compara el desempeño de diferentes modelos de difusión (por ejemplo, IC y LT) usando validación cruzada.
Permite elegir el modelo que mejor se ajusta a los datos reales.

🕸️ Visualización de Red de Propagación
--------------------------------------
Un grafo interactivo que muestra cómo se conecta la información entre usuarios durante la propagación.
Permite visualizar la estructura de la difusión: quiénes son los nodos clave, cuántos niveles hay, etc.


MANUAL DE USO RÁPIDO
--------------------
1. Inicia sesión como admin o usuario.
2. Carga o selecciona un dataset.
3. Ejecuta el análisis.
4. Explora los resultados en los cuadros y gráficas del dashboard.
5. (Admin) Consulta estadísticas y usuarios registrados en la sección de administración.


NOTA FINAL
----------
Este sistema es una herramienta educativa y de análisis para entender la propagación de información en redes sociales usando grafos y métodos estadísticos avanzados. Puedes personalizar los datasets y explorar diferentes escenarios de difusión.
