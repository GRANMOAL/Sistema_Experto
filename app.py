from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import joblib
import numpy as np
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_validate, train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, f1_score, make_scorer, precision_score, recall_score
import json

app = Flask(__name__)
app.secret_key = 'pe_causas'  # Cambiar por una clave segura

# Configuración para subida de archivos
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Crear carpeta de uploads si no existe
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('flask_app/models', exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Función para cargar modelos de forma segura
def load_models():
    models = {}
    model_paths = {
        'id3': "flask_app/models/modelo_id3_hardware.joblib",
        'knn': "flask_app/models/modelo_knn_hardware.joblib"
    }
    
    for model_name, path in model_paths.items():
        try:
            if os.path.exists(path):
                models[model_name] = joblib.load(path)
                print(f"✓ Modelo {model_name.upper()} cargado correctamente")
            else:
                print(f"⚠ Archivo no encontrado: {path}")
        except Exception as e:
            print(f"✗ Error cargando modelo {model_name}: {str(e)}")
    
    return models

# Cargar modelos
models = load_models()

# Lista de columnas que el usuario debe llenar
columns = ['MYCT', 'MMAX', 'CACH', 'CHMAX', 'ERP']

# Descripciones de las columnas
column_descriptions = {
    'MYCT': 'Tiempo de ciclo de máquina (nanosegundos)',
    'MMAX': 'Memoria principal máxima (kilobytes)',
    'CACH': 'Memoria caché (kilobytes)',
    'CHMAX': 'Canales máximos',
    'ERP': 'Rendimiento relativo estimado'
}

def validate_input(user_data):
    """Valida los datos de entrada del usuario"""
    errors = []
    
    for col in columns:
        if col not in user_data or user_data[col].strip() == '':
            errors.append(f"El campo {col} es obligatorio")
    
    if errors:
        return False, errors
    
    try:
        values = []
        for col in columns:
            value = float(user_data[col])
            if col in ['MYCT', 'MMAX', 'CACH', 'CHMAX'] and value < 0:
                errors.append(f"{col} debe ser un valor positivo")
            values.append(value)
            
    except ValueError as e:
        errors.append("Todos los campos deben contener números válidos")
        return False, errors
    
    if errors:
        return False, errors
    
    return True, values

def make_prediction(model_name, input_data):
    """Realiza la predicción usando el modelo seleccionado"""
    if model_name not in models:
        return f"Error: Modelo {model_name.upper()} no disponible"
    
    try:
        input_array = np.array([input_data])
        prediction = models[model_name].predict(input_array)[0]
        
        if isinstance(prediction, (int, float)):
            return f"Predicción ({model_name.upper()}): {prediction:.2f}"
        else:
            return f"Predicción ({model_name.upper()}): {prediction}"
            
    except Exception as e:
        return f"Error en la predicción: {str(e)}"

# ===== RUTAS PRINCIPALES =====

@app.route("/")
def home():
    """Página principal con enlaces a las diferentes funcionalidades"""
    return render_template("home.html")

# ===== GUI 1: ENTRENAMIENTO DE MODELOS =====

@app.route("/train", methods=["GET", "POST"])
def train_model():
    """GUI 1: Cargar CSV y entrenar modelos"""
    if request.method == "POST":
        try:
            # Verificar si se subió un archivo
            if 'csv_file' not in request.files:
                flash('No se seleccionó ningún archivo', 'error')
                return redirect(request.url)
            
            file = request.files['csv_file']
            if file.filename == '':
                flash('No se seleccionó ningún archivo', 'error')
                return redirect(request.url)
            
            if file and allowed_file(file.filename):
                # Guardar archivo
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Leer CSV
                df = pd.read_csv(filepath)
                
                # Obtener parámetros del formulario
                model_type = request.form.get('model_type')
                validation_method = request.form.get('validation_method')
                
                # Preparar datos (asumiendo que la última columna es el target)
                X = df.iloc[:, :-1]
                y = df.iloc[:, -1]
                
                results = {}
                
                if model_type == 'id3':
                    # Entrenar modelo ID3 (Decision Tree)
                    model = DecisionTreeClassifier(criterion='entropy', random_state=42)
                    
                    if validation_method == 'holdout':
                        test_size = float(request.form.get('test_size', 0.3))
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calcular métricas
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Determinar el tipo de promedio para métricas multiclase
                        unique_classes = len(np.unique(y))
                        avg_method = 'weighted' if unique_classes > 2 else 'binary'
                        
                        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                        
                        # Reporte de clasificación detallado
                        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        
                        results = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'method': 'Hold-out',
                            'test_size': test_size,
                            'model_type': 'ID3',
                            'classification_report': class_report,
                            'num_classes': unique_classes,
                            'avg_method': avg_method
                        }
                        
                    elif validation_method == 'kfold':
                        k_folds = int(request.form.get('k_folds', 5))
                        
                        # Definir scoring para cross-validation
                        unique_classes = len(np.unique(y))
                        avg_method = 'weighted' if unique_classes > 2 else 'binary'
                        
                        scoring = {
                            'accuracy': 'accuracy',
                            'precision': make_scorer(precision_score, average=avg_method, zero_division=0),
                            'recall': make_scorer(recall_score, average=avg_method, zero_division=0),
                            'f1': make_scorer(f1_score, average=avg_method, zero_division=0)
                        }
                        
                        cv_results = cross_validate(model, X, y, cv=k_folds, scoring=scoring)
                        
                        model.fit(X, y)  # Entrenar con todos los datos
                        
                        results = {
                            'accuracy': cv_results['test_accuracy'].mean(),
                            'accuracy_std': cv_results['test_accuracy'].std(),
                            'precision': cv_results['test_precision'].mean(),
                            'precision_std': cv_results['test_precision'].std(),
                            'recall': cv_results['test_recall'].mean(),
                            'recall_std': cv_results['test_recall'].std(),
                            'f1_score': cv_results['test_f1'].mean(),
                            'f1_score_std': cv_results['test_f1'].std(),
                            'method': f'{k_folds}-Fold Cross Validation',
                            'model_type': 'ID3',
                            'k_folds': k_folds,
                            'num_classes': unique_classes,
                            'avg_method': avg_method,
                            'cv_scores': {
                                'accuracy': cv_results['test_accuracy'].tolist(),
                                'precision': cv_results['test_precision'].tolist(),
                                'recall': cv_results['test_recall'].tolist(),
                                'f1': cv_results['test_f1'].tolist()
                            }
                        }
                
                elif model_type == 'knn':
                    # Entrenar modelo KNN
                    n_neighbors = int(request.form.get('n_neighbors', 5))
                    model = KNeighborsClassifier(n_neighbors=n_neighbors)
                    
                    if validation_method == 'holdout':
                        test_size = float(request.form.get('test_size', 0.3))
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=42
                        )
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        
                        # Calcular métricas
                        accuracy = accuracy_score(y_test, y_pred)
                        
                        # Determinar el tipo de promedio para métricas multiclase
                        unique_classes = len(np.unique(y))
                        avg_method = 'weighted' if unique_classes > 2 else 'binary'
                        
                        precision = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
                        recall = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
                        f1 = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
                        
                        # Reporte de clasificación detallado
                        class_report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
                        
                        results = {
                            'accuracy': accuracy,
                            'precision': precision,
                            'recall': recall,
                            'f1_score': f1,
                            'method': 'Hold-out',
                            'test_size': test_size,
                            'model_type': 'KNN',
                            'n_neighbors': n_neighbors,
                            'classification_report': class_report,
                            'num_classes': unique_classes,
                            'avg_method': avg_method
                        }
                        
                    elif validation_method == 'kfold':
                        k_folds = int(request.form.get('k_folds', 5))
                        
                        # Definir scoring para cross-validation
                        unique_classes = len(np.unique(y))
                        avg_method = 'weighted' if unique_classes > 2 else 'binary'
                        
                        scoring = {
                            'accuracy': 'accuracy',
                            'precision': make_scorer(precision_score, average=avg_method, zero_division=0),
                            'recall': make_scorer(recall_score, average=avg_method, zero_division=0),
                            'f1': make_scorer(f1_score, average=avg_method, zero_division=0)
                        }
                        
                        cv_results = cross_validate(model, X, y, cv=k_folds, scoring=scoring)
                        
                        model.fit(X, y)  # Entrenar con todos los datos
                        
                        results = {
                            'accuracy': cv_results['test_accuracy'].mean(),
                            'accuracy_std': cv_results['test_accuracy'].std(),
                            'precision': cv_results['test_precision'].mean(),
                            'precision_std': cv_results['test_precision'].std(),
                            'recall': cv_results['test_recall'].mean(),
                            'recall_std': cv_results['test_recall'].std(),
                            'f1_score': cv_results['test_f1'].mean(),
                            'f1_score_std': cv_results['test_f1'].std(),
                            'method': f'{k_folds}-Fold Cross Validation',
                            'model_type': 'KNN',
                            'n_neighbors': n_neighbors,
                            'k_folds': k_folds,
                            'num_classes': unique_classes,
                            'avg_method': avg_method,
                            'cv_scores': {
                                'accuracy': cv_results['test_accuracy'].tolist(),
                                'precision': cv_results['test_precision'].tolist(),
                                'recall': cv_results['test_recall'].tolist(),
                                'f1': cv_results['test_f1'].tolist()
                            }
                        }
                
                # Guardar modelo entrenado
                model_filename = f"modelo_{model_type}_hardware.joblib"
                model_path = os.path.join("flask_app/models", model_filename)
                joblib.dump(model, model_path)
                
                # Actualizar modelos cargados en memoria
                models[model_type] = model
                
                flash(f'Modelo {model_type.upper()} entrenado exitosamente!', 'success')
                return render_template("train.html", results=results)
            
            else:
                flash('Tipo de archivo no permitido. Solo archivos CSV.', 'error')
                
        except Exception as e:
            flash(f'Error durante el entrenamiento: {str(e)}', 'error')
    
    return render_template("train.html")

# ===== GUI 2: PREDICCIÓN =====

@app.route("/predict", methods=["GET", "POST"])
def predict():
    """GUI 2: Realizar predicciones con modelos entrenados"""
    prediction = None
    errors = []
    user_values = {}
    
    if request.method == "POST":
        selected_model = request.form.get("model")
        
        if not selected_model:
            errors.append("Por favor selecciona un modelo")
        elif selected_model not in ['id3', 'knn']:
            errors.append("Modelo seleccionado no válido")
        
        user_values = {col: request.form.get(col, '') for col in columns}
        
        if not errors:
            is_valid, validation_result = validate_input(user_values)
            
            if is_valid:
                prediction = make_prediction(selected_model, validation_result)
            else:
                errors = validation_result
    
    available_models = list(models.keys())
    
    return render_template(
        "predict.html", 
        columns=columns,
        column_descriptions=column_descriptions,
        prediction=prediction,
        errors=errors,
        user_values=user_values,
        available_models=available_models
    )

# ===== RUTAS AUXILIARES =====

@app.route("/info")
def info():
    """Información sobre los modelos y columnas"""
    model_info = {
        'id3': 'Árbol de decisión ID3 - Bueno para datos categóricos y reglas claras',
        'knn': 'K-Nearest Neighbors - Efectivo para patrones locales en los datos'
    }
    
    return render_template(
        "info.html",
        columns=columns,
        column_descriptions=column_descriptions,
        model_info=model_info,
        available_models=list(models.keys())
    )

@app.route("/models/status")
def models_status():
    """API endpoint para verificar el estado de los modelos"""
    return jsonify({
        'available_models': list(models.keys()),
        'total_models': len(models)
    })

if __name__ == "__main__":
    print("\n" + "="*50)
    print("🚀 INICIANDO APLICACIÓN FLASK")
    print("="*50)
    
    if not models:
        print("⚠ ADVERTENCIA: No se cargaron modelos. Usa la GUI de entrenamiento.")
    else:
        print(f"✓ {len(models)} modelo(s) disponible(s): {', '.join(models.keys())}")
    print("="*50 + "\n")
    
    app.run(debug=True)