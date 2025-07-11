from flask import Flask, jsonify, request, render_template_string, redirect, url_for
from flask_cors import CORS
import json
import threading
import webbrowser
import time
from propagacion_analyzer import PropagacionAnalyzer
import numpy as np
import jwt
import hashlib
from datetime import datetime, timedelta
from functools import wraps
import os
import numpy as pd

app = Flask(__name__)
CORS(app)

# Configuración de seguridad
app.config['SECRET_KEY'] = 'tu_clave_secreta_aqui_cambiar_en_produccion'
app.config['JWT_EXPIRATION_DELTA'] = timedelta(hours=24)

# Usuarios del sistema (en producción usar base de datos)
USUARIOS = {
    'admin': {
        'password': hashlib.sha256('admin123'.encode()).hexdigest(),
        'role': 'admin',
        'name': 'Administrador'
    },
    'analyst': {
        'password': hashlib.sha256('analyst123'.encode()).hexdigest(),
        'role': 'analyst',
        'name': 'Analista'
    }
}

# Variable global para el analizador
analyzer = None
analysis_status = {
    'is_running': False,
    'progress': 0,
    'current_step': '',
    'results': None
}

def verificar_token(token):
    """Verifica si el token JWT es válido"""
    try:
        payload = jwt.decode(token, app.config['SECRET_KEY'], algorithms=['HS256'])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def requiere_auth(f):
    """Decorador para rutas que requieren autenticación"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        token = None
        
        # Buscar token en headers
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            try:
                token = auth_header.split(' ')[1]  # Bearer TOKEN
            except IndexError:
                return jsonify({'success': False, 'message': 'Token malformado'}), 401
        
        if not token:
            return jsonify({'success': False, 'message': 'Token requerido'}), 401
        
        payload = verificar_token(token)
        if not payload:
            return jsonify({'success': False, 'message': 'Token inválido o expirado'}), 401
        
        # Agregar información del usuario al request
        request.current_user = payload
        return f(*args, **kwargs)
    
    return decorated_function
#--------------------------------------------------
@app.route('/admin-dashboard')
def admin_dashboard():
    """Servir el dashboard de administrador"""
    try:
        with open('admin_dashboard.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inyectar token de autenticación (igual que en el dashboard normal)
        auth_script = """
        <script>
        // Verificar autenticación al cargar
        const token = localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
        if (!token) {
            window.location.href = '/login';
        } else {
            // Verificar rol de admin
            fetch('/api/verify-admin', {
                headers: {
                    'Authorization': 'Bearer ' + token
                }
            })
            .then(response => response.json())
            .then(data => {
                if (!data.success || data.role !== 'admin') {
                    window.location.href = '/';
                }
            });
            
            // Agregar token a todas las peticiones fetch
            const originalFetch = window.fetch;
            window.fetch = function(...args) {
                if (args[1]) {
                    args[1].headers = args[1].headers || {};
                    args[1].headers['Authorization'] = 'Bearer ' + token;
                } else {
                    args[1] = {
                        headers: {
                            'Authorization': 'Bearer ' + token
                        }
                    };
                }
                return originalFetch.apply(this, args);
            };
        }
        </script>
        """
        
        html_content = html_content.replace('</body>', auth_script + '</body>')
        return html_content
    except FileNotFoundError:
        return "Dashboard de administrador no encontrado", 404
    


@app.route('/api/verify-admin')
@requiere_auth
def verify_admin():
    """Verificar si el usuario es administrador"""
    user_info = request.current_user
    return jsonify({
        'success': True,
        'role': user_info['role'],
        'is_admin': user_info['role'] == 'admin'
    })
#-------------------------------------------------
@app.route('/')
def dashboard():
    """Servir el dashboard HTML - la autenticación se maneja en el frontend"""
    try:
        with open('dashboard_propagacion.html', 'r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Inyectar token de autenticación en el HTML
        auth_script = """
        <script>
        // Verificar autenticación al cargar
        const token = localStorage.getItem('auth_token') || sessionStorage.getItem('auth_token');
        if (!token) {
            window.location.href = '/login';
        } else {
            // Agregar token a todas las peticiones fetch
            const originalFetch = window.fetch;
            window.fetch = function(...args) {
                if (args[1]) {
                    args[1].headers = args[1].headers || {};
                    args[1].headers['Authorization'] = 'Bearer ' + token;
                } else {
                    args[1] = {
                        headers: {
                            'Authorization': 'Bearer ' + token
                        }
                    };
                }
                return originalFetch.apply(this, args);
            };
        }
        </script>
        """
        
        # Insertar script antes del cierre del body
        html_content = html_content.replace('</body>', auth_script + '</body>')
        
        return html_content
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Error</title>
        </head>
        <body>
            <h1>Error: No se encontró el archivo dashboard_propagacion.html</h1>
            <p>Asegúrate de que el archivo del dashboard esté en el mismo directorio.</p>
        </body>
        </html>
        """

@app.route('/login')
def login_page():
    """Servir página de login"""
    try:
        with open('login.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Login</title>
        </head>
        <body>
            <h1>Error: No se encontró el archivo login.html</h1>
            <p>Crea el archivo login.html en el mismo directorio.</p>
        </body>
        </html>
        """

@app.route('/api/login', methods=['POST'])
def login():
    """Endpoint para autenticación"""
    try:
        datos = request.get_json()
        username = datos.get('username')
        password = datos.get('password')
        remember = datos.get('remember', False)
        
        if not username or not password:
            return jsonify({
                'success': False,
                'message': 'Usuario y contraseña son requeridos'
            }), 400
        
        # Verificar usuario
        if username not in USUARIOS:
            return jsonify({
                'success': False,
                'message': 'Usuario no encontrado'
            }), 401
        
        # Verificar contraseña
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        if USUARIOS[username]['password'] != password_hash:
            return jsonify({
                'success': False,
                'message': 'Contraseña incorrecta'
            }), 401
        
        # Generar token JWT
        expiration = datetime.utcnow() + (timedelta(days=7) if remember else timedelta(hours=24))
        payload = {
            'username': username,
            'role': USUARIOS[username]['role'],
            'name': USUARIOS[username]['name'],
            'exp': expiration,
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
        
        return jsonify({
            'success': True,
            'token': token,
            'user': {
                'username': username,
                'role': USUARIOS[username]['role'],
                'name': USUARIOS[username]['name']
            },
            'message': 'Login exitoso'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error en login: {str(e)}'
        }), 500

@app.route('/api/logout', methods=['POST'])
@requiere_auth
def logout():
    """Endpoint para cerrar sesión"""
    return jsonify({
        'success': True,
        'message': 'Sesión cerrada exitosamente'
    })

@app.route('/api/profile')
@requiere_auth
def profile():
    """Obtener información del usuario actual"""
    return jsonify({
        'success': True,
        'user': request.current_user
    })

@app.route('/api/run-analysis', methods=['POST'])
@requiere_auth
def run_analysis():
    """Ejecutar análisis completo de propagación - requiere autenticación"""
    global analyzer, analysis_status
    
    if analysis_status['is_running']:
        return jsonify({
            'success': False,
            'message': 'Ya hay un análisis en ejecución'
        })
    
    try:
        # Obtener parámetros
        params = request.get_json()
        time_window = params.get('time_window', 24)
        bootstrap_samples = params.get('bootstrap_samples', 100)
        cv_folds = params.get('cv_folds', 3)
        dataset_type = params.get('dataset_type', 'twitter')
        
        # Log de usuario que ejecuta análisis
        user_info = request.current_user
        print(f"Usuario {user_info['username']} ({user_info['role']}) ejecutando análisis")
        
        # Iniciar análisis en hilo separado
        thread = threading.Thread(
            target=execute_analysis,
            args=(time_window, bootstrap_samples, cv_folds, dataset_type)
        )
        thread.start()
        
        return jsonify({
            'success': True,
            'message': 'Análisis iniciado exitosamente',
            'user': user_info['name']
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error iniciando análisis: {str(e)}'
        })

# Todas las rutas API existentes necesitan el decorador @requiere_auth
@app.route('/api/status')
@requiere_auth
def get_status():
    """Obtener estado actual del análisis"""
    return jsonify(analysis_status)

@app.route('/api/results')
@requiere_auth
def get_results():
    """Obtener resultados del último análisis"""
    if analysis_status['results']:
        return jsonify({
            'success': True,
            'data': analysis_status['results']
        })
    else:
        return jsonify({
            'success': False,
            'message': 'No hay resultados disponibles'
        })

@app.route('/api/export-results')
@requiere_auth
def export_results():
    """Exportar resultados en formato JSON"""
    if analysis_status['results']:
        return jsonify(analysis_status['results'])
    else:
        return jsonify({
            'error': 'No hay resultados para exportar'
        })

@app.route('/api/cascade/<int:cascade_id>')
@requiere_auth
def get_cascade_details(cascade_id):
    """Obtener detalles de una cascada específica"""
    global analyzer
    
    if not analyzer or cascade_id >= len(analyzer.cascadas):
        return jsonify({
            'success': False,
            'message': 'Cascada no encontrada'
        })
    
    cascada = analyzer.cascadas[cascade_id]
    metricas = analyzer.calcular_metricas_viralidad(cascada)
    
    return jsonify({
        'success': True,
        'data': {
            'cascada': {
                'nodo_inicial': cascada['nodo_inicial'],
                'secuencia': cascada['secuencia'],
                'alcance_total': cascada['alcance_total'],
                'velocidad_propagacion': float(cascada['velocidad_propagacion']),
                'profundidad_maxima': cascada['profundidad_maxima']
            },
            'metricas': {
                metrica: float(valor)
                for metrica, valor in metricas.items()
            }
        }
    })
#---------------------------------------------------------
# En backend_integration.py, agregar esta ruta (ya existe pero la mejoraremos)
@app.route('/api/upload-dataset', methods=['POST'])
@requiere_auth
def upload_dataset():
    """Endpoint para subir datasets personalizados"""
    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No se encontró el archivo'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'Nombre de archivo vacío'}), 400
        
        if not file.filename.endswith('.csv'):
            return jsonify({'success': False, 'message': 'Solo se aceptan archivos CSV'}), 400
        
        # Crear directorio data si no existe
        os.makedirs('data', exist_ok=True)
        
        # Validar estructura del CSV antes de guardar
        try:
            # Leer solo las primeras líneas para validar
            df = pd.read_csv(file.stream)
            required_columns = {'user_id', 'follower_id', 'timestamp'}
            if not required_columns.issubset(df.columns):
                missing = required_columns - set(df.columns)
                return jsonify({
                    'success': False,
                    'message': f'El CSV debe contener las columnas: {required_columns}. Faltan: {missing}'
                }), 400
            
            # Volver al inicio del archivo después de la validación
            file.stream.seek(0)
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Error validando el CSV: {str(e)}'
            }), 400
        
        # Guardar archivo
        filename = f"data/personalizado_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        file.save(filename)
        
        return jsonify({
            'success': True,
            'message': 'Archivo subido exitosamente',
            'filename': filename,
            'columns': list(df.columns),
            'preview': df.head(5).to_dict('records')
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Error al subir archivo: {str(e)}'
        }), 500
#----------------------------------------------------------
# Resto de funciones existentes (execute_analysis, compile_results, etc.)
def execute_analysis(time_window, bootstrap_samples, cv_folds, dataset_type):
    """Ejecutar el análisis completo"""
    global analyzer, analysis_status
    
    try:
        analysis_status['is_running'] = True
        analysis_status['progress'] = 0
        analysis_status['current_step'] = 'Inicializando...'
        
        # Crear analizador
        analyzer = PropagacionAnalyzer()
        analysis_status['progress'] = 10
        analysis_status['current_step'] = 'Verificando datasets...'
        
        # Verificar y preparar datasets
        analyzer.verificar_y_preparar_datasets()
        
        analysis_status['progress'] = 15
        analysis_status['current_step'] = f'Cargando datos {dataset_type}...'
        
        # Cargar datos según el tipo seleccionado
        df = None
        if dataset_type == "twitter":
            df = analyzer.cargar_datos_twitter_cascade()
        elif dataset_type == "digg":
            df = analyzer.cargar_datos_digg()
        elif dataset_type == "memetracker":
            df = analyzer.cargar_datos_memetracker()
        elif dataset_type.startswith("data/personalizado_"):  # Dataset personalizado
            df = analyzer.cargar_datos_personalizado(dataset_type, {
                'user_id': 'user_id',
                'follower_id': 'follower_id',
                'timestamp': 'timestamp'
            })
        
        # Verificar que se cargaron datos correctamente
        if df is None or len(df) == 0:
            raise ValueError(f"No se pudieron cargar datos para el dataset tipo: {dataset_type}")
        
        if analyzer.grafo.number_of_nodes() == 0:
            raise ValueError("El grafo no tiene nodos. Verifica el formato de los datos.")
        
        analysis_status['progress'] = 25
        analysis_status['current_step'] = 'Detectando cascadas...'
        
        # Detectar cascadas - usar nodos con más conexiones como semilla
        nodos_disponibles = list(analyzer.grafo.nodes())
        
        # Seleccionar nodos con mayor grado como semillas
        grados = [(nodo, analyzer.grafo.degree(nodo)) for nodo in nodos_disponibles]
        grados.sort(key=lambda x: x[1], reverse=True)
        
        # Usar top nodos o mínimo 15
        num_nodos_semilla = min(15, len(nodos_disponibles))
        nodos_iniciales = [nodo for nodo, grado in grados[:num_nodos_semilla]]
        
        print(f"Detectando cascadas desde {len(nodos_iniciales)} nodos semilla...")
        
        cascadas_detectadas = 0
        for i, nodo in enumerate(nodos_iniciales):
            try:
                cascada = analyzer.detectar_cascadas(nodo, ventana_tiempo=time_window)
                if cascada and cascada['alcance_total'] > 1:
                    cascadas_detectadas += 1
                    print(f"Cascada {cascadas_detectadas}: Nodo {nodo} -> {cascada['alcance_total']} nodos")
            except Exception as e:
                print(f"Error detectando cascada desde nodo {nodo}: {e}")
                
            analysis_status['progress'] = 25 + (i / len(nodos_iniciales)) * 25
        
        if cascadas_detectadas == 0:
            raise ValueError("No se detectaron cascadas válidas. Verifica los datos o ajusta la ventana temporal.")
        
        print(f"Total de cascadas detectadas: {cascadas_detectadas}")
        
        analysis_status['progress'] = 50
        analysis_status['current_step'] = 'Ejecutando bootstrap temporal...'
        
        # Bootstrap temporal
        bootstrap_results, intervalos_confianza = analyzer.bootstrap_temporal(
            n_bootstrap=bootstrap_samples
        )
        
        if bootstrap_results is None:
            raise ValueError("Error en bootstrap temporal")
        
        analysis_status['progress'] = 70
        analysis_status['current_step'] = 'Ejecutando validación cruzada...'
        
        # Validación cruzada temporal
        cv_results, cv_stats = analyzer.validacion_cruzada_temporal(n_folds=cv_folds)
        
        if cv_results is None:
            print("Advertencia: No se pudo completar la validación cruzada")
            cv_results = {}
            cv_stats = {}
        
        analysis_status['progress'] = 85
        analysis_status['current_step'] = 'Analizando robustez de modelos...'
        
        # Análisis de robustez
        modelos_results, robustez_scores = analyzer.analizar_robustez_modelos()
        
        analysis_status['progress'] = 95
        analysis_status['current_step'] = 'Compilando resultados...'
        
        # Compilar resultados finales
        results = compile_results(
            analyzer, bootstrap_results, intervalos_confianza,
            cv_results, cv_stats, modelos_results, robustez_scores
        )
        
        analysis_status['results'] = results
        analysis_status['progress'] = 100
        analysis_status['current_step'] = 'Análisis completado'
        
        print(f"✅ Análisis completado con éxito para dataset {dataset_type}")
        print(f"   - Cascadas detectadas: {cascadas_detectadas}")
        print(f"   - Nodos en el grafo: {analyzer.grafo.number_of_nodes()}")
        print(f"   - Aristas en el grafo: {analyzer.grafo.number_of_edges()}")

    except Exception as e:
        error_msg = f'Error en análisis: {str(e)}'
        print(f"❌ {error_msg}")
        analysis_status['current_step'] = error_msg
        analysis_status['results'] = None
        import traceback
        traceback.print_exc()
    finally:
        analysis_status['is_running'] = False

def compile_results(analyzer, bootstrap_results, intervalos_confianza, 
                   cv_results, cv_stats, modelos_results, robustez_scores):
    """Compilar todos los resultados en formato JSON"""
    
    # Función para convertir valores numpy a python nativo
    def convert_to_native(val):
        if isinstance(val, (np.integer, np.floating)):
            return float(val) if isinstance(val, np.floating) else int(val)
        elif isinstance(val, dict):
            return {k: convert_to_native(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [convert_to_native(v) for v in val]
        return val
    
    # Métricas básicas
    cascadas = analyzer.cascadas
    alcances = [c['alcance_total'] for c in cascadas]
    velocidades = [c['velocidad_propagacion'] for c in cascadas]
    
    # Métricas de viralidad promedio
    metricas_viralidad_promedio = {}
    for cascada in cascadas:
        metricas = analyzer.calcular_metricas_viralidad(cascada)
        for metrica, valor in metricas.items():
            if metrica not in metricas_viralidad_promedio:
                metricas_viralidad_promedio[metrica] = []
            metricas_viralidad_promedio[metrica].append(valor)
    
    for metrica in metricas_viralidad_promedio:
        valores = metricas_viralidad_promedio[metrica]
        metricas_viralidad_promedio[metrica] = {
            'media': float(np.mean(valores)),
            'std': float(np.std(valores)),
            'min': float(np.min(valores)),
            'max': float(np.max(valores))
        }
    
    # Convertir todos los valores numpy a nativos antes de retornar
    results = {
        'basic_metrics': {
            'total_cascades': len(cascadas),
            'total_nodes': analyzer.grafo.number_of_nodes(),
            'total_edges': analyzer.grafo.number_of_edges(),
            'avg_reach': float(np.mean(alcances)) if alcances else 0,
            'max_reach': int(np.max(alcances)) if alcances else 0,
            'avg_velocity': float(np.mean(velocidades)) if velocidades else 0,
            'max_velocity': float(np.max(velocidades)) if velocidades else 0
        },
        'cascades_data': [
            {
                'id': i,
                'nodo_inicial': c['nodo_inicial'],
                'alcance_total': c['alcance_total'],
                'velocidad_propagacion': float(c['velocidad_propagacion']),
                'profundidad_maxima': c['profundidad_maxima']
            }
            for i, c in enumerate(cascadas)
        ],
        'bootstrap_results': {
            'intervalos_confianza': {
                metrica: {
                    'media': float(stats['media']),
                    'std': float(stats['std']),
                    'ic_inferior': float(stats['ic_inferior']),
                    'ic_superior': float(stats['ic_superior'])
                }
                for metrica, stats in intervalos_confianza.items()
            },
            'velocidades_bootstrap': [float(v) for v in bootstrap_results['velocidades']],
            'alcances_bootstrap': [float(a) for a in bootstrap_results['alcances']]
        },
        'cv_results': {
            modelo: {
                metrica: float(valores['media']) if isinstance(valores, dict) else float(valores)
                for metrica, valores in stats.items()
            }
            for modelo, stats in cv_stats.items()
        } if cv_stats else {},
        'robustez_scores': {
            modelo: float(score)
            for modelo, score in robustez_scores.items()
        },
        'viralidad_metrics': metricas_viralidad_promedio
    }
    
    return convert_to_native(results)

def open_browser():
    """Abrir navegador después de un breve delay"""
    time.sleep(1.5)
    webbrowser.open('http://localhost:5000/login')

def create_dashboard_interface():
    """
    Crear interfaz completa del dashboard con autenticación
    """
    
    print("="*60)
    print("🚀 INICIANDO DASHBOARD DE PROPAGACIÓN CON AUTENTICACIÓN")
    print("="*60)
    print()
    print("🔐 Sistema de Login:")
    print("   ✅ Autenticación JWT")
    print("   ✅ Sesiones persistentes")
    print("   ✅ Rutas protegidas")
    print()
    print("📊 Funcionalidades disponibles:")
    print("   ✅ Análisis de cascadas en tiempo real")
    print("   ✅ Bootstrap temporal con IC 95%")
    print("   ✅ Validación cruzada temporal")
    print("   ✅ Análisis de robustez de modelos")
    print("   ✅ Visualizaciones interactivas")
    print("   ✅ Exportación de resultados")
    print()
    print("🌐 Acceso al dashboard:")
    print("   URL: http://localhost:5000/login")
    print("   Usuario demo: admin / admin123")
    print("   El navegador se abrirá automáticamente...")
    print()
    print("🔧 Para detener el servidor: Ctrl+C")
    print("="*60)
    
    # Abrir navegador en hilo separado
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    # Iniciar servidor Flask
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\n\n🛑 Servidor detenido por el usuario")
    except Exception as e:
        print(f"\n❌ Error en el servidor: {e}")


if __name__ == "__main__":
    create_dashboard_interface()