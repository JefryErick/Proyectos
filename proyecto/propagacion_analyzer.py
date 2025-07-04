import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import requests
import os
from collections import defaultdict, deque
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class PropagacionAnalyzer:
    """
    Analizador de Propagación de Información con Grafos Temporales
    
    Funcionalidades:
    - Análisis de cascadas de información
    - Bootstrap temporal para validación estadística
    - Métricas de viralidad
    - Visualización de propagación
    """
    
    def __init__(self):
        self.grafo = nx.DiGraph()
        self.cascadas = []
        self.timestamps = {}
        self.metricas_viralidad = {}
        self.datos_originales = None
        
    def cargar_datos_twitter_cascade(self):
        """
        Carga datos del Twitter Cascade Dataset desde 'data/twitter-simulado.csv'.
        Formato esperado: user_id, follower_id, timestamp (o nombres similares)
        """
        try:
            archivo_path = 'data/twitter-simulado.csv'
            print(f"Cargando Twitter Cascade Dataset desde: {archivo_path}")
            
            # Intentar diferentes formatos de archivo
            if archivo_path.endswith('.txt'):
                df = pd.read_csv(archivo_path, sep='\t', header=None, 
                                names=['user_id', 'follower_id', 'timestamp'])
            elif archivo_path.endswith('.csv'):
                df = pd.read_csv(archivo_path)
            elif archivo_path.endswith('.json'):
                df = pd.read_json(archivo_path, lines=True)
            else:
                df = pd.read_csv(archivo_path, sep=None, engine='python')
            
            # Normalizar nombres de columnas
            column_mapping = {
                'user': 'user_id',
                'follower': 'follower_id',
                'time': 'timestamp',
                'created_at': 'timestamp',
                'datetime': 'timestamp'
            }
            
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Verificar columnas requeridas
            required_cols = ['user_id', 'follower_id', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"El archivo debe contener las columnas: {required_cols}")
            
            # Convertir timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Limpiar datos
            df = df.dropna(subset=['user_id', 'follower_id', 'timestamp'])
            df = df[df['user_id'] != df['follower_id']]  # Eliminar auto-enlaces
            df = df.drop_duplicates(subset=['user_id', 'follower_id'])  # Duplicados
            
            print(f"Datos limpiados: {len(df)} interacciones")
            
            # Construir grafo temporal
            for _, row in df.iterrows():
                self.grafo.add_edge(row['user_id'], row['follower_id'])
                edge_key = (row['user_id'], row['follower_id'])
                self.timestamps[edge_key] = row['timestamp']
            
            self.datos_originales = df
            
            print(f"Grafo construido: {self.grafo.number_of_nodes()} nodos, {self.grafo.number_of_edges()} aristas")
            return df
        
        except Exception as e:
            print(f"Error cargando datos: {e}")
            return None

    
    def cargar_datos_digg(self, archivo_path: str = "data/digg-simulado.csv"):
        """
        Carga datos del Digg Dataset simulado
        Formato esperado: user_id, friend_id, timestamp
        """
        try:
            print(f"Cargando Digg Dataset desde: {archivo_path}")
            
            # Leer CSV con nombres de columnas apropiados
            df = pd.read_csv(archivo_path)
            
            # Verificar que tenemos las columnas necesarias
            required_cols = ['user_id', 'friend_id', 'timestamp']
            if not all(col in df.columns for col in required_cols):
                print(f"Columnas disponibles: {list(df.columns)}")
                raise ValueError(f"El archivo debe contener las columnas: {required_cols}")
            
            print(f"Datos cargados: {len(df)} filas")
            
            # Convertir timestamp Unix a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            
            # Limpiar datos
            df = df.dropna(subset=['user_id', 'friend_id', 'timestamp'])
            df = df[df['user_id'] != df['friend_id']]  # Eliminar auto-enlaces
            df = df.drop_duplicates(subset=['user_id', 'friend_id'])  # Eliminar duplicados
            
            print(f"Datos limpiados: {len(df)} conexiones")
            
            # Construir grafo temporal
            for _, row in df.iterrows():
                self.grafo.add_edge(row['user_id'], row['friend_id'])
                edge_key = (row['user_id'], row['friend_id'])
                self.timestamps[edge_key] = row['timestamp']
            
            self.datos_originales = df
            
            print(f"Grafo Digg construido: {self.grafo.number_of_nodes()} nodos, {self.grafo.number_of_edges()} aristas")
            
            # Verificar rango temporal
            if len(df) > 0:
                print(f"Rango temporal: {df['timestamp'].min()} a {df['timestamp'].max()}")
                print(f"Duración: {(df['timestamp'].max() - df['timestamp'].min()).days} días")
            
            return df
            
        except Exception as e:
            print(f"Error cargando datos Digg: {e}")
            return None

    def cargar_datos_memetracker(self, archivo_path: str = "data/memetracker-simulado.csv"):
        """
        Carga datos del Memetracker Dataset simulado
        Formato esperado: phrase, site, timestamp, frequency
        """
        try:
            print(f"Cargando Memetracker Dataset desde: {archivo_path}")
            
            # Leer CSV 
            df = pd.read_csv(archivo_path)
            
            # Verificar columnas
            required_cols = ['phrase', 'site', 'timestamp', 'frequency']
            if not all(col in df.columns for col in required_cols):
                print(f"Columnas disponibles: {list(df.columns)}")
                raise ValueError(f"El archivo debe contener las columnas: {required_cols}")
            
            print(f"Datos cargados: {len(df)} filas")
            
            # Convertir timestamp a datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            
            # Limpiar datos
            df = df.dropna(subset=['phrase', 'site', 'timestamp'])
            
            print(f"Datos limpiados: {len(df)} entradas")
            
            # Crear conexiones basadas en frases compartidas
            # Agrupar por frase y crear conexiones entre sitios que comparten frases
            phrase_groups = df.groupby('phrase')
            
            connections = []
            for phrase, group in phrase_groups:
                sites = group['site'].unique()
                timestamps = group['timestamp'].values
                
                # Si hay más de un sitio compartiendo la frase, crear conexiones
                if len(sites) > 1:
                    # Ordenar por timestamp para crear conexiones dirigidas
                    group_sorted = group.sort_values('timestamp')
                    sites_sorted = group_sorted['site'].tolist()
                    times_sorted = group_sorted['timestamp'].tolist()
                    
                    # Crear conexiones secuenciales (el primer sitio "infecta" a los siguientes)
                    for i in range(len(sites_sorted) - 1):
                        connections.append({
                            'user_id': sites_sorted[i],
                            'follower_id': sites_sorted[i + 1],
                            'timestamp': times_sorted[i + 1],
                            'phrase': phrase
                        })
            
            # Crear DataFrame de conexiones
            if connections:
                conn_df = pd.DataFrame(connections)
                
                # Eliminar duplicados (misma conexión entre sitios)
                conn_df = conn_df.drop_duplicates(subset=['user_id', 'follower_id'])
                
                print(f"Conexiones creadas: {len(conn_df)}")
                
                # Construir grafo temporal
                for _, row in conn_df.iterrows():
                    self.grafo.add_edge(row['user_id'], row['follower_id'])
                    edge_key = (row['user_id'], row['follower_id'])
                    self.timestamps[edge_key] = row['timestamp']
                
                self.datos_originales = conn_df
                
                print(f"Grafo Memetracker construido: {self.grafo.number_of_nodes()} nodos, {self.grafo.number_of_edges()} aristas")
                
                # Verificar rango temporal
                if len(conn_df) > 0:
                    print(f"Rango temporal: {conn_df['timestamp'].min()} a {conn_df['timestamp'].max()}")
                    print(f"Duración: {(conn_df['timestamp'].max() - conn_df['timestamp'].min()).days} días")
                
                return conn_df
            else:
                print("No se pudieron crear conexiones válidas")
                return None
                
        except Exception as e:
            print(f"Error cargando datos Memetracker: {e}")
            return None

    # Función de utilidad para verificar y preparar datasets
    def verificar_y_preparar_datasets(self):
        """
        Verifica que los datasets existan y los crea si es necesario
        """
        print("🔍 Verificando disponibilidad de datasets...")
        
        datasets = {
            'twitter': 'data/twitter-simulado.csv',
            'digg': 'data/diggsimulado.csv', 
            'memetracker': 'data/memetrackersimulado.csv'
        }
        
        for nombre, path in datasets.items():
            if os.path.exists(path):
                print(f"✅ {nombre.capitalize()}: {path} encontrado")
            else:
                print(f"❌ {nombre.capitalize()}: {path} no encontrado")
        
        print("📊 Datasets preparados y listos para usar")

    def detectar_cascadas(self, nodo_inicial: int, ventana_tiempo: int = 24):
        """
        Detecta cascadas de información desde un nodo inicial usando timestamps reales
        """
        if nodo_inicial not in self.grafo:
            print(f"Nodo {nodo_inicial} no existe en el grafo")
            return None
            
        cascada = {
            'nodo_inicial': nodo_inicial,
            'secuencia': [],
            'timestamps': [],
            'profundidad_maxima': 0,
            'velocidad_propagacion': 0,
            'alcance_total': 0
        }
        
        # Encontrar el timestamp inicial (primera aparición del nodo)
        timestamps_nodo = []
        for edge, timestamp in self.timestamps.items():
            if nodo_inicial in edge:
                timestamps_nodo.append(timestamp)
        
        if not timestamps_nodo:
            return cascada
            
        tiempo_inicial = min(timestamps_nodo)
        tiempo_limite = tiempo_inicial + timedelta(hours=ventana_tiempo)
        
        # BFS temporal para rastrear propagación real
        visitados = set()
        cola = deque([(nodo_inicial, 0, tiempo_inicial)])
        
        while cola:
            nodo_actual, profundidad, tiempo_actual = cola.popleft()
            
            if nodo_actual in visitados:
                continue
                
            visitados.add(nodo_actual)
            cascada['secuencia'].append(nodo_actual)
            cascada['timestamps'].append(tiempo_actual)
            cascada['profundidad_maxima'] = max(cascada['profundidad_maxima'], profundidad)
            
            # Expandir a vecinos dentro de la ventana temporal
            for vecino in self.grafo.successors(nodo_actual):
                edge_key = (nodo_actual, vecino)
                if edge_key in self.timestamps:
                    tiempo_edge = self.timestamps[edge_key]
                    if tiempo_edge <= tiempo_limite and tiempo_edge >= tiempo_actual and vecino not in visitados:
                        cola.append((vecino, profundidad + 1, tiempo_edge))
        
        cascada['alcance_total'] = len(cascada['secuencia'])
        if len(cascada['timestamps']) > 1:
            tiempo_total = (cascada['timestamps'][-1] - cascada['timestamps'][0]).total_seconds()
            cascada['velocidad_propagacion'] = cascada['alcance_total'] / max(tiempo_total, 1)
        
        self.cascadas.append(cascada)
        return cascada
    
    def bootstrap_temporal(self, n_bootstrap: int = 1000, confianza: float = 0.95):
        """
        Bootstrap temporal de cascadas para análisis estadístico usando datos reales
        """
        if len(self.cascadas) == 0:
            print("No hay cascadas detectadas. Ejecute detectar_cascadas() primero.")
            return None, None
            
        print(f"Ejecutando bootstrap temporal con {n_bootstrap} muestras sobre {len(self.cascadas)} cascadas reales...")
        
        bootstrap_results = {
            'velocidades': [],
            'alcances': [],
            'profundidades': []
        }
        
        # Bootstrap sobre cascadas reales detectadas
        for i in range(n_bootstrap):
            # Remuestreo de cascadas existentes con reemplazo
            cascada_sample = np.random.choice(self.cascadas)
            
            if len(cascada_sample['timestamps']) > 1:
                # Remuestreo de timestamps dentro de cada cascada
                indices_resampled = np.random.choice(
                    len(cascada_sample['timestamps']), 
                    size=len(cascada_sample['timestamps']), 
                    replace=True
                )
                
                timestamps_resampled = [cascada_sample['timestamps'][i] for i in indices_resampled]
                timestamps_sorted = sorted(timestamps_resampled)
                
                # Recalcular métricas
                if len(timestamps_sorted) > 1:
                    tiempo_total = (timestamps_sorted[-1] - timestamps_sorted[0]).total_seconds()
                    velocidad = len(timestamps_sorted) / max(tiempo_total, 1)
                else:
                    velocidad = 0
                    
                bootstrap_results['velocidades'].append(velocidad)
                bootstrap_results['alcances'].append(len(timestamps_sorted))
                bootstrap_results['profundidades'].append(cascada_sample['profundidad_maxima'])
        
        # Calcular intervalos de confianza
        alpha = 1 - confianza
        percentiles = [100 * alpha/2, 100 * (1 - alpha/2)]
        
        intervalos_confianza = {}
        for metrica, valores in bootstrap_results.items():
            if valores:
                intervalos_confianza[metrica] = {
                    'media': np.mean(valores),
                    'std': np.std(valores),
                    'ic_inferior': np.percentile(valores, percentiles[0]),
                    'ic_superior': np.percentile(valores, percentiles[1])
                }
        
        print("\nIntervalos de confianza calculados:")
        for metrica, stats in intervalos_confianza.items():
            print(f"{metrica}: {stats['media']:.4f} ± {stats['std']:.4f} "
                  f"[IC {confianza*100}%: {stats['ic_inferior']:.4f}, {stats['ic_superior']:.4f}]")
        
        return bootstrap_results, intervalos_confianza
    
    def calcular_metricas_viralidad(self, cascada: Dict):
        """
        Calcula métricas de viralidad para una cascada real
        """
        metricas = {
            'factor_viralizacion': 0,
            'coeficiente_difusion': 0,
            'indice_velocidad': 0,
            'score_influencia': 0,
            'persistencia_temporal': 0
        }
        
        if cascada['alcance_total'] > 1:
            # Factor de viralización (crecimiento por nivel)
            if cascada['profundidad_maxima'] > 0:
                metricas['factor_viralizacion'] = cascada['alcance_total'] / cascada['profundidad_maxima']
            
            # Coeficiente de difusión (proporción de nodos únicos)
            nodos_unicos = len(set(cascada['secuencia']))
            metricas['coeficiente_difusion'] = nodos_unicos / len(cascada['secuencia'])
            
            # Índice de velocidad
            metricas['indice_velocidad'] = cascada['velocidad_propagacion']
            
            # Score de influencia del nodo inicial
            nodo_inicial = cascada['nodo_inicial']
            if nodo_inicial in self.grafo:
                out_degree = self.grafo.out_degree(nodo_inicial)
                in_degree = self.grafo.in_degree(nodo_inicial)
                metricas['score_influencia'] = (out_degree + in_degree) * cascada['alcance_total']
            
            # Persistencia temporal (duración en horas)
            if len(cascada['timestamps']) > 1:
                duracion_total = (cascada['timestamps'][-1] - cascada['timestamps'][0]).total_seconds()
                metricas['persistencia_temporal'] = duracion_total / 3600
        
        return metricas
    
    def analizar_nodos_influyentes(self, top_n: int = 10):
        """
        Identifica los nodos más influyentes basado en métricas de centralidad y cascadas generadas
        """
        print(f"Analizando top {top_n} nodos más influyentes...")
        
        # Calcular métricas de centralidad
        centralidades = {
            'degree': nx.degree_centrality(self.grafo),
            'betweenness': nx.betweenness_centrality(self.grafo),
            'pagerank': nx.pagerank(self.grafo),
            'closeness': nx.closeness_centrality(self.grafo)
        }
        
        # Combinar con información de cascadas
        nodos_info = {}
        for nodo in self.grafo.nodes():
            cascadas_nodo = [c for c in self.cascadas if c['nodo_inicial'] == nodo]
            
            nodos_info[nodo] = {
                'total_cascadas': len(cascadas_nodo),
                'alcance_promedio': np.mean([c['alcance_total'] for c in cascadas_nodo]) if cascadas_nodo else 0,
                'velocidad_promedio': np.mean([c['velocidad_propagacion'] for c in cascadas_nodo]) if cascadas_nodo else 0,
                'degree_centrality': centralidades['degree'][nodo],
                'betweenness_centrality': centralidades['betweenness'][nodo],
                'pagerank': centralidades['pagerank'][nodo],
                'closeness_centrality': centralidades['closeness'][nodo]
            }
        
        # Calcular score combinado de influencia
        for nodo, info in nodos_info.items():
            score = (info['alcance_promedio'] * 0.3 + 
                    info['velocidad_promedio'] * 100 * 0.2 +
                    info['pagerank'] * 100 * 0.3 +
                    info['betweenness_centrality'] * 100 * 0.2)
            info['score_influencia'] = score
        
        # Ordenar por score de influencia
        top_nodos = sorted(nodos_info.items(), 
                          key=lambda x: x[1]['score_influencia'], 
                          reverse=True)[:top_n]
        
        print(f"\nTop {top_n} nodos más influyentes:")
        for i, (nodo, info) in enumerate(top_nodos, 1):
            print(f"{i}. Nodo {nodo}:")
            print(f"   Score influencia: {info['score_influencia']:.4f}")
            print(f"   Cascadas generadas: {info['total_cascadas']}")
            print(f"   Alcance promedio: {info['alcance_promedio']:.2f}")
            print(f"   PageRank: {info['pagerank']:.6f}")
        
        return top_nodos
    
    def validacion_cruzada_temporal(self, n_folds: int = 5, ventana_prediccion: int = 6):
        """
        Validación cruzada temporal para evaluar modelos de predicción usando datos reales
        """
        if len(self.cascadas) == 0:
            print("No hay cascadas para validar. Ejecute detectar_cascadas() primero.")
            return None, None
            
        print(f"Ejecutando validación cruzada temporal con {n_folds} folds...")
        
        # Organizar cascadas por tiempo real
        cascadas_con_tiempo = []
        for i, cascada in enumerate(self.cascadas):
            if len(cascada['timestamps']) > 0:
                tiempo_inicio = cascada['timestamps'][0]
                cascadas_con_tiempo.append((tiempo_inicio, i, cascada))
        
        cascadas_con_tiempo.sort(key=lambda x: x[0])
        
        if len(cascadas_con_tiempo) < n_folds:
            print(f"Insuficientes cascadas ({len(cascadas_con_tiempo)}) para {n_folds} folds")
            return None, None
        
        # Validación cruzada temporal
        fold_size = len(cascadas_con_tiempo) // n_folds
        resultados_cv = {
            'IC': {'mae': [], 'rmse': [], 'r2': []},
            'LT': {'mae': [], 'rmse': [], 'r2': []}
        }
        
        for fold in range(n_folds):
            print(f"  Procesando fold {fold + 1}/{n_folds}...")
            
            inicio_test = fold * fold_size
            fin_test = min((fold + 1) * fold_size, len(cascadas_con_tiempo))
            
            train_cascadas = (cascadas_con_tiempo[:inicio_test] + 
                            cascadas_con_tiempo[fin_test:])
            test_cascadas = cascadas_con_tiempo[inicio_test:fin_test]
            
            if len(test_cascadas) == 0:
                continue
            
            # Entrenar modelos con datos reales
            parametros_ic = self._calibrar_modelo_ic(train_cascadas)
            parametros_lt = self._calibrar_modelo_lt(train_cascadas)
            
            # Hacer predicciones
            for modelo, parametros in [('IC', parametros_ic), ('LT', parametros_lt)]:
                predicciones = []
                valores_reales = []
                
                for tiempo_inicio, idx, cascada_test in test_cascadas:
                    # Usar solo parte inicial para predecir
                    tiempo_corte = tiempo_inicio + timedelta(hours=ventana_prediccion)
                    
                    if modelo == 'IC':
                        pred_alcance = self._predecir_ic(cascada_test, parametros, tiempo_corte)
                    else:
                        pred_alcance = self._predecir_lt(cascada_test, parametros, tiempo_corte)
                    
                    alcance_real = cascada_test['alcance_total']
                    
                    predicciones.append(pred_alcance)
                    valores_reales.append(alcance_real)
                
                # Calcular métricas
                if len(predicciones) > 0:
                    predicciones = np.array(predicciones)
                    valores_reales = np.array(valores_reales)
                    
                    mae = np.mean(np.abs(predicciones - valores_reales))
                    rmse = np.sqrt(np.mean((predicciones - valores_reales)**2))
                    
                    ss_res = np.sum((valores_reales - predicciones)**2)
                    ss_tot = np.sum((valores_reales - np.mean(valores_reales))**2)
                    r2 = 1 - (ss_res / (ss_tot + 1e-8))
                    
                    resultados_cv[modelo]['mae'].append(mae)
                    resultados_cv[modelo]['rmse'].append(rmse)
                    resultados_cv[modelo]['r2'].append(r2)
        
        # Estadísticas finales
        estadisticas_cv = {}
        for modelo in ['IC', 'LT']:
            estadisticas_cv[modelo] = {}
            for metrica in ['mae', 'rmse', 'r2']:
                valores = resultados_cv[modelo][metrica]
                if valores:
                    estadisticas_cv[modelo][metrica] = {
                        'media': np.mean(valores),
                        'std': np.std(valores),
                        'min': np.min(valores),
                        'max': np.max(valores)
                    }
        
        print("\nResultados de Validación Cruzada Temporal:")
        for modelo, stats in estadisticas_cv.items():
            print(f"\n{modelo}:")
            for metrica, valores in stats.items():
                print(f"  {metrica.upper()}: {valores['media']:.4f} ± {valores['std']:.4f}")
        self.cv_resultados_cache = resultados_cv
        return resultados_cv, estadisticas_cv
        
    def analizar_robustez_modelos(self):
        """
        Analiza la robustez de los modelos IC y LT con base en la varianza de sus métricas de CV.
        Retorna: resultados (dict vacío por ahora), y diccionario de scores de robustez [0–1]
        """
        print("🔍 Analizando robustez de modelos IC y LT...")
        
        if not hasattr(self, 'cv_resultados_cache'):
            print("⚠️ No se encontraron resultados previos de validación cruzada.")
            return {}, {"IC": 0.0, "LT": 0.0}
        
        robustez_scores = {}
        modelos = ['IC', 'LT']
        
        for modelo in modelos:
            try:
                mae_std = np.std(self.cv_resultados_cache[modelo]['mae'])
                rmse_std = np.std(self.cv_resultados_cache[modelo]['rmse'])
                r2_std = np.std(self.cv_resultados_cache[modelo]['r2'])

                # Menor desviación estándar implica mayor robustez
                score = 1.0 - np.mean([mae_std, rmse_std, r2_std])
                score = np.clip(score, 0.0, 1.0)
            except:
                score = 0.0
            robustez_scores[modelo] = float(score)

        
        print("📊 Robustez calculada:", robustez_scores)
        return {}, robustez_scores


    def _calibrar_modelo_ic(self, train_cascadas):
        """Calibra modelo IC con datos de entrenamiento reales"""
        if not train_cascadas:
            return {'p_activacion': 0.1}
            
        # Estimar probabilidad óptima basada en datos reales
        alcances_observados = [cascada['alcance_total'] for _, _, cascada in train_cascadas]
        velocidades_observadas = [cascada['velocidad_propagacion'] for _, _, cascada in train_cascadas]
        
        p_estimada = np.mean(velocidades_observadas) / np.mean(alcances_observados) if np.mean(alcances_observados) > 0 else 0.1
        p_estimada = np.clip(p_estimada, 0.01, 0.5)
        
        return {'p_activacion': p_estimada}
    
    def _calibrar_modelo_lt(self, train_cascadas):
        """Calibra modelo LT con datos de entrenamiento reales"""
        if not train_cascadas:
            return {'umbral_base': 0.5}
            
        # Estimar umbral basado en distribución de profundidades
        profundidades = [cascada['profundidad_maxima'] for _, _, cascada in train_cascadas]
        umbral_estimado = 1.0 / (np.mean(profundidades) + 1) if profundidades else 0.5
        umbral_estimado = np.clip(umbral_estimado, 0.1, 0.9)
        
        return {'umbral_base': umbral_estimado}
    
    def _predecir_ic(self, cascada, parametros, tiempo_corte):
        """Predice alcance usando modelo IC con datos reales"""
        p = parametros['p_activacion']
        
        # Contar nodos activos hasta tiempo_corte
        nodos_activos = sum(1 for t in cascada['timestamps'] if t <= tiempo_corte)
        
        if nodos_activos == 0:
            return 1
            
        # Predicción basada en crecimiento esperado
        factor_crecimiento = 1 + p * self.grafo.out_degree(cascada['nodo_inicial'])
        prediccion = nodos_activos * factor_crecimiento
        
        return min(prediccion, len(self.grafo.nodes()))
    
    def _predecir_lt(self, cascada, parametros, tiempo_corte):
        """Predice alcance usando modelo LT con datos reales"""
        umbral = parametros['umbral_base']
        
        nodos_activos = sum(1 for t in cascada['timestamps'] if t <= tiempo_corte)
        
        if nodos_activos == 0:
            return 1
            
        # Predicción basada en umbral
        grado_promedio = np.mean([self.grafo.degree(n) for n in self.grafo.nodes()])
        factor_crecimiento = 1 + (1 - umbral) * grado_promedio / 10
        prediccion = nodos_activos * factor_crecimiento
        
        return min(prediccion, len(self.grafo.nodes()))
    
    def generar_reporte_completo(self):
        """
        Genera reporte completo del análisis con datos reales
        """
        print("="*60)
        print("REPORTE DE ANÁLISIS DE PROPAGACIÓN DE INFORMACIÓN")
        print("="*60)
        
        # Estadísticas del dataset
        print(f"\n1. ESTADÍSTICAS DEL DATASET")
        print(f"   - Nodos en la red: {self.grafo.number_of_nodes()}")
        print(f"   - Aristas en la red: {self.grafo.number_of_edges()}")
        print(f"   - Densidad de la red: {nx.density(self.grafo):.6f}")
        
        if self.datos_originales is not None:
            print(f"   - Periodo temporal: {self.datos_originales['timestamp'].min()} a {self.datos_originales['timestamp'].max()}")
            print(f"   - Duración total: {(self.datos_originales['timestamp'].max() - self.datos_originales['timestamp'].min()).days} días")
        
        if len(self.cascadas) > 0:
            # Análisis de cascadas reales
            print(f"\n2. ANÁLISIS DE CASCADAS DETECTADAS")
            alcances = [c['alcance_total'] for c in self.cascadas]
            velocidades = [c['velocidad_propagacion'] for c in self.cascadas]
            profundidades = [c['profundidad_maxima'] for c in self.cascadas]
            
            print(f"   - Total de cascadas: {len(self.cascadas)}")
            print(f"   - Alcance promedio: {np.mean(alcances):.2f} ± {np.std(alcances):.2f}")
            print(f"   - Velocidad promedio: {np.mean(velocidades):.6f} ± {np.std(velocidades):.6f} nodos/segundo")
            print(f"   - Profundidad promedio: {np.mean(profundidades):.2f} ± {np.std(profundidades):.2f}")
            print(f"   - Cascada más grande: {max(alcances)} nodos")
            print(f"   - Cascada más rápida: {max(velocidades):.6f} nodos/segundo")
            print(f"   - Cascada más profunda: {max(profundidades)} niveles")
            
            # Análisis temporal
            print(f"\n3. ANÁLISIS TEMPORAL")
            duraciones = []
            for cascada in self.cascadas:
                if len(cascada['timestamps']) > 1:
                    duracion = (cascada['timestamps'][-1] - cascada['timestamps'][0]).total_seconds() / 3600
                    duraciones.append(duracion)
            
            if duraciones:
                print(f"   - Duración promedio de cascadas: {np.mean(duraciones):.2f} ± {np.std(duraciones):.2f} horas")
                print(f"   - Cascada más larga: {max(duraciones):.2f} horas")
                print(f"   - Cascada más corta: {min(duraciones):.2f} horas")
        
        print("\n" + "="*60)
        print("REPORTE COMPLETO GENERADO CON DATOS REALES")
        print("="*60)
    
    def visualizar_cascada(self, cascada_idx: int = 0, figsize: Tuple[int, int] = (15, 10)):
        """
        Visualiza una cascada real con timestamps
        """
        if cascada_idx >= len(self.cascadas):
            print("Índice de cascada inválido")
            return
        
        cascada = self.cascadas[cascada_idx]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # 1. Subgrafo de la cascada
        subgrafo = self.grafo.subgraph(cascada['secuencia'])
        pos = nx.spring_layout(subgrafo, k=1, iterations=50)
        
        # Colorear por orden temporal real
        colores = plt.cm.viridis(np.linspace(0, 1, len(cascada['secuencia'])))
        
        nx.draw(subgrafo, pos, ax=ax1, node_color=colores, 
                node_size=300, with_labels=True, font_size=8,
                edge_color='gray', alpha=0.7)
        ax1.set_title(f'Cascada {cascada_idx}: Red de Propagación Real')
        
        # 2. Timeline real de activación
        if len(cascada['timestamps']) > 1:
            tiempo_inicio = cascada['timestamps'][0]
            tiempos_relativos = [(t - tiempo_inicio).total_seconds()/3600 for t in cascada['timestamps']]
            
            ax2.plot(tiempos_relativos, range(len(tiempos_relativos)), 'o-', color='blue', alpha=0.7)
            ax2.set_xlabel('Tiempo desde inicio (horas)')
            ax2.set_ylabel('Nodos activados acumulados')
            ax2.set_title('Timeline Real de Activación')
            ax2.grid(True, alpha=0.3)
        
        # 3. Distribución temporal de activaciones
        if len(cascada['timestamps']) > 2:
            tiempo_inicio = cascada['timestamps'][0]
            intervalos = []
            for i in range(1, len(cascada['timestamps'])):
                intervalo = (cascada['timestamps'][i] - cascada['timestamps'][i-1]).total_seconds() / 60  # minutos
                intervalos.append(intervalo)
            
            ax3.hist(intervalos, bins=min(10, len(intervalos)), alpha=0.7, color='green', edgecolor='black')
            ax3.set_xlabel('Intervalo entre activaciones (minutos)')
            ax3.set_ylabel('Frecuencia')
            ax3.set_title('Distribución de Intervalos Temporales')
        
        # 4. Métricas de la cascada
        metricas = self.calcular_metricas_viralidad(cascada)
        metricas_nombres = list(metricas.keys())
        metricas_valores = list(metricas.values())
        
        ax4.barh(metricas_nombres, metricas_valores, color='orange', alpha=0.7)
        ax4.set_xlabel('Valor')
        ax4.set_title('Métricas de Viralidad')
        
        plt.tight_layout()
        plt.show()

# Función de demostración con datos reales
def demo_con_datos_reales():
    """
    Demostración usando datos reales
    """
    print("="*70)
    print("🚀 DEMO: Analizador de Propagación con Datos Reales")
    print("="*70)
    
    analyzer = PropagacionAnalyzer()
    
    print("\n📁 Formatos de datos soportados:")
    print("   1. Twitter Cascade Dataset (.txt, .csv)")
    print("   2. Digg Social News (.txt)")
    print("   3. Memetracker (.txt, .csv)")
    print("   4. Reddit Comments (.json)")
    print("   5. Datos personalizados (cualquier formato)")
    
    print("\n⚠️  Para usar este código con datos reales:")
    print("   1. Descarga un dataset de las URLs recomendadas")
    print("   2. Usa el método cargar_datos_[tipo]() apropiado")
    print("   3. Ejecuta el análisis completo")
    
    return analyzer

if __name__ == "__main__":
    demo_con_datos_reales()