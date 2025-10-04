import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from typing import Dict, Tuple, Any, Optional, List
import json

# Imports condicionais para ML (caso não estejam instalados)
try:
    from imblearn.over_sampling import SMOTE
    from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import (classification_report, roc_auc_score, accuracy_score, 
                                roc_curve, confusion_matrix, mean_squared_error, r2_score)
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, LinearRegression
    from sklearn.svm import SVC, SVR
    from sklearn.base import BaseEstimator
    from xgboost import XGBClassifier, XGBRegressor
    ML_AVAILABLE = True
except ImportError as e:
    ML_AVAILABLE = False
    print(f"Algumas bibliotecas de ML não estão disponíveis: {e}")


class MLToolsError(Exception):
    """Exceção personalizada para ferramentas de ML"""
    pass


class DataPreprocessor:
    """Classe para pré-processamento de dados"""
    
    def __init__(self):
        self.scaler = None
        self.label_encoders = {}
        self.target_encoder = None
        
    def analyze_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analisa a qualidade dos dados"""
        try:
            analysis = {
                "shape": df.shape,
                "missing_values": df.isnull().sum().to_dict(),
                "missing_percentage": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
                "data_types": df.dtypes.astype(str).to_dict(),
                "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": df.select_dtypes(include=['object', 'category']).columns.tolist(),
                "duplicated_rows": int(df.duplicated().sum()),
                "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2)
            }
            
            # Estatísticas para colunas numéricas
            if analysis["numeric_columns"]:
                numeric_stats = df[analysis["numeric_columns"]].describe().to_dict()
                analysis["numeric_stats"] = numeric_stats
            
            # Contagem de valores únicos para categóricas
            if analysis["categorical_columns"]:
                categorical_stats = {}
                for col in analysis["categorical_columns"]:
                    categorical_stats[col] = {
                        "unique_values": int(df[col].nunique()),
                        "most_frequent": str(df[col].mode().iloc[0]) if not df[col].empty else None
                    }
                analysis["categorical_stats"] = categorical_stats
            
            return analysis
            
        except Exception as e:
            raise MLToolsError(f"Erro na análise de qualidade dos dados: {str(e)}")
    
    def prepare_data_for_ml(self, df: pd.DataFrame, target_column: str, 
                          test_size: float = 0.2, balance_data: bool = True) -> Dict[str, Any]:
        """Prepara dados para machine learning"""
        if not ML_AVAILABLE:
            raise MLToolsError("Bibliotecas de ML não estão disponíveis")
            
        try:
            # Verificar se a coluna target existe
            if target_column not in df.columns:
                raise MLToolsError(f"Coluna target '{target_column}' não encontrada")
            
            # Separar features e target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # Identificar problema (classificação ou regressão)
            is_classification = (y.dtype == 'object' or y.nunique() <= 10)
            
            # Encode categorical variables
            X_processed = X.copy()
            
            # Encode features categóricas
            categorical_cols = X_processed.select_dtypes(include=['object']).columns
            for col in categorical_cols:
                le = LabelEncoder()
                X_processed[col] = le.fit_transform(X_processed[col].astype(str))
                self.label_encoders[col] = le
            
            # Encode target se for classificação
            if is_classification and y.dtype == 'object':
                self.target_encoder = LabelEncoder()
                y = self.target_encoder.fit_transform(y)
            
            # Split dos dados
            X_train, X_test, y_train, y_test = train_test_split(
                X_processed, y, test_size=test_size, random_state=42,
                stratify=y if is_classification else None
            )
            
            # Escalonamento
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Balanceamento (apenas para classificação)
            if is_classification and balance_data and len(np.unique(y_train)) > 1:
                try:
                    smote = SMOTE(random_state=42)
                    X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
                except Exception:
                    # Se SMOTE falhar, usar dados originais
                    X_train_balanced, y_train_balanced = X_train_scaled, y_train
            else:
                X_train_balanced, y_train_balanced = X_train_scaled, y_train
            
            return {
                "X_train": X_train_balanced,
                "y_train": y_train_balanced,
                "X_test": X_test_scaled,
                "y_test": y_test,
                "is_classification": is_classification,
                "feature_names": X.columns.tolist(),
                "original_shapes": {
                    "X_train": X_train.shape,
                    "X_test": X_test.shape,
                    "after_balancing": X_train_balanced.shape
                },
                "class_distribution": pd.Series(y_train_balanced).value_counts().to_dict() if is_classification else None
            }
            
        except Exception as e:
            raise MLToolsError(f"Erro no pré-processamento: {str(e)}")


class ModelTrainer:
    """Classe para treinamento de modelos"""
    
    def __init__(self):
        self.trained_models = {}
        
    def get_default_models(self, is_classification: bool = True) -> Dict[str, BaseEstimator]:
        """Retorna modelos padrão baseado no tipo de problema"""
        if not ML_AVAILABLE:
            raise MLToolsError("Bibliotecas de ML não estão disponíveis")
            
        if is_classification:
            return {
                "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
                "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
                "XGBoost": XGBClassifier(random_state=42, eval_metric='logloss'),
                "SVM": SVC(random_state=42, probability=True)
            }
        else:
            return {
                "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
                "Linear Regression": LinearRegression(),
                "XGBoost": XGBRegressor(random_state=42),
                "SVM": SVR()
            }
    
    def train_multiple_models(self, X_train, y_train, X_test, y_test, 
                            is_classification: bool = True) -> Dict[str, Any]:
        """Treina múltiplos modelos e retorna resultados"""
        try:
            models = self.get_default_models(is_classification)
            results = {}
            
            for name, model in models.items():
                try:
                    # Treinamento
                    model.fit(X_train, y_train)
                    
                    # Predições
                    y_pred = model.predict(X_test)
                    
                    if is_classification:
                        # Métricas de classificação
                        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else y_pred
                        
                        report = classification_report(y_test, y_pred, output_dict=True)
                        
                        metrics = {
                            "accuracy": accuracy_score(y_test, y_pred),
                            "precision": report.get('weighted avg', {}).get('precision', 0),
                            "recall": report.get('weighted avg', {}).get('recall', 0),
                            "f1_score": report.get('weighted avg', {}).get('f1-score', 0),
                            "roc_auc": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) == 2 else None
                        }
                    else:
                        # Métricas de regressão
                        metrics = {
                            "mse": mean_squared_error(y_test, y_pred),
                            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
                            "r2_score": r2_score(y_test, y_pred),
                            "mae": np.mean(np.abs(y_test - y_pred))
                        }
                    
                    results[name] = {
                        "model": model,
                        "predictions": y_pred,
                        "metrics": metrics
                    }
                    
                    # Salvar modelo treinado
                    self.trained_models[name] = model
                    
                except Exception as e:
                    results[name] = {"error": str(e)}
            
            return results
            
        except Exception as e:
            raise MLToolsError(f"Erro no treinamento dos modelos: {str(e)}")
    
    def optimize_model(self, model_name: str, X_train, y_train, 
                      is_classification: bool = True, n_iter: int = 20) -> Dict[str, Any]:
        """Otimiza hiperparâmetros de um modelo específico"""
        if not ML_AVAILABLE:
            raise MLToolsError("Bibliotecas de ML não estão disponíveis")
            
        try:
            # Definir parâmetros baseado no modelo
            if "xgboost" in model_name.lower():
                if is_classification:
                    base_model = XGBClassifier(random_state=42, eval_metric='logloss')
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2],
                        'subsample': [0.8, 0.9, 1.0],
                        'colsample_bytree': [0.8, 0.9, 1.0]
                    }
                else:
                    base_model = XGBRegressor(random_state=42)
                    param_grid = {
                        'n_estimators': [100, 200, 300],
                        'max_depth': [3, 5, 7],
                        'learning_rate': [0.01, 0.1, 0.2]
                    }
            
            elif "random forest" in model_name.lower():
                if is_classification:
                    base_model = RandomForestClassifier(random_state=42)
                else:
                    base_model = RandomForestRegressor(random_state=42)
                    
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            
            else:
                raise MLToolsError(f"Otimização não implementada para {model_name}")
            
            # Executar busca
            scoring = 'roc_auc' if is_classification else 'r2'
            random_search = RandomizedSearchCV(
                base_model, param_grid, n_iter=n_iter, scoring=scoring,
                cv=3, random_state=42, n_jobs=-1
            )
            
            random_search.fit(X_train, y_train)
            
            return {
                "best_params": random_search.best_params_,
                "best_score": random_search.best_score_,
                "best_model": random_search.best_estimator_
            }
            
        except Exception as e:
            raise MLToolsError(f"Erro na otimização: {str(e)}")


class ModelEvaluator:
    """Classe para avaliação e visualização de modelos"""
    
    def create_comparison_plots(self, results: Dict[str, Any], y_test, 
                               is_classification: bool = True) -> Tuple[plt.Figure, plt.Figure]:
        """Cria gráficos de comparação de modelos"""
        try:
            if is_classification:
                return self._create_classification_plots(results, y_test)
            else:
                return self._create_regression_plots(results, y_test)
        except Exception as e:
            raise MLToolsError(f"Erro na criação de gráficos: {str(e)}")
    
    def _create_classification_plots(self, results, y_test):
        """Cria gráficos para problemas de classificação"""
        # Preparar dados para gráfico de barras
        comparison_data = []
        for model_name, result in results.items():
            if "error" not in result:
                metrics = result["metrics"]
                comparison_data.append({
                    "Model": model_name,
                    "Accuracy": metrics["accuracy"],
                    "Precision": metrics["precision"],
                    "Recall": metrics["recall"],
                    "F1-Score": metrics["f1_score"]
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Gráfico de barras
        fig_bar, ax_bar = plt.subplots(figsize=(12, 6))
        metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1-Score"]
        
        df_melted = df_comparison.melt(
            id_vars="Model", 
            value_vars=metrics_to_plot,
            var_name="Metric", 
            value_name="Score"
        )
        
        sns.barplot(data=df_melted, x="Metric", y="Score", hue="Model", ax=ax_bar)
        ax_bar.set_title("Comparação de Performance dos Modelos de Classificação")
        ax_bar.set_ylim(0, 1.1)
        ax_bar.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Gráfico de Curva ROC
        fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
        
        for model_name, result in results.items():
            if "error" not in result and result["metrics"]["roc_auc"] is not None:
                model = result["model"]
                if hasattr(model, 'predict_proba'):
                    y_prob = model.predict_proba(results[list(results.keys())[0]].get("X_test", []))
                    if len(y_prob.shape) > 1 and y_prob.shape[1] > 1:
                        y_prob = y_prob[:, 1]
                    fpr, tpr, _ = roc_curve(y_test, y_prob)
                    auc_score = result["metrics"]["roc_auc"]
                    ax_roc.plot(fpr, tpr, label=f"{model_name} (AUC={auc_score:.3f})")
        
        ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
        ax_roc.set_xlabel("Taxa de Falsos Positivos")
        ax_roc.set_ylabel("Taxa de Verdadeiros Positivos")
        ax_roc.set_title("Comparação das Curvas ROC")
        ax_roc.legend()
        ax_roc.grid(True, alpha=0.3)
        
        return fig_bar, fig_roc
    
    def _create_regression_plots(self, results, y_test):
        """Cria gráficos para problemas de regressão"""
        # Gráfico de comparação de métricas
        comparison_data = []
        for model_name, result in results.items():
            if "error" not in result:
                metrics = result["metrics"]
                comparison_data.append({
                    "Model": model_name,
                    "R² Score": metrics["r2_score"],
                    "RMSE": metrics["rmse"],
                    "MAE": metrics["mae"]
                })
        
        df_comparison = pd.DataFrame(comparison_data)
        
        # Gráfico de métricas
        fig_metrics, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # R² Score
        sns.barplot(data=df_comparison, x="Model", y="R² Score", ax=ax1)
        ax1.set_title("Comparação R² Score")
        ax1.tick_params(axis='x', rotation=45)
        
        # RMSE e MAE
        df_melted = df_comparison.melt(
            id_vars="Model",
            value_vars=["RMSE", "MAE"],
            var_name="Metric",
            value_name="Value"
        )
        sns.barplot(data=df_melted, x="Model", y="Value", hue="Metric", ax=ax2)
        ax2.set_title("Comparação RMSE vs MAE")
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Gráfico de predição vs real
        fig_pred, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for idx, (model_name, result) in enumerate(results.items()):
            if "error" not in result and idx < 4:
                y_pred = result["predictions"]
                ax = axes[idx]
                
                ax.scatter(y_test, y_pred, alpha=0.6)
                ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                ax.set_xlabel("Valores Reais")
                ax.set_ylabel("Predições")
                ax.set_title(f"{model_name}\nR² = {result['metrics']['r2_score']:.3f}")
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        return fig_metrics, fig_pred
    
    def generate_model_summary(self, results: Dict[str, Any], is_classification: bool = True) -> Dict[str, Any]:
        """Gera resumo dos resultados dos modelos"""
        try:
            summary = {
                "total_models": len(results),
                "successful_models": len([r for r in results.values() if "error" not in r]),
                "failed_models": len([r for r in results.values() if "error" in r]),
                "model_rankings": [],
                "recommendations": []
            }
            
            # Ranking dos modelos
            valid_results = {k: v for k, v in results.items() if "error" not in v}
            
            if is_classification:
                ranking_metric = "accuracy"
                sorted_models = sorted(
                    valid_results.items(),
                    key=lambda x: x[1]["metrics"][ranking_metric],
                    reverse=True
                )
            else:
                ranking_metric = "r2_score"
                sorted_models = sorted(
                    valid_results.items(),
                    key=lambda x: x[1]["metrics"][ranking_metric],
                    reverse=True
                )
            
            for rank, (model_name, result) in enumerate(sorted_models, 1):
                summary["model_rankings"].append({
                    "rank": rank,
                    "model": model_name,
                    "score": result["metrics"][ranking_metric]
                })
            
            # Recomendações
            if sorted_models:
                best_model = sorted_models[0]
                summary["recommendations"].append(
                    f"🏆 Melhor modelo: {best_model[0]} com {ranking_metric} = {best_model[1]['metrics'][ranking_metric]:.3f}"
                )
                
                if len(sorted_models) > 1:
                    second_best = sorted_models[1]
                    summary["recommendations"].append(
                        f"🥈 Segunda opção: {second_best[0]} com {ranking_metric} = {second_best[1]['metrics'][ranking_metric]:.3f}"
                    )
            
            return summary
            
        except Exception as e:
            raise MLToolsError(f"Erro na geração do resumo: {str(e)}")


# Funções de conveniência para uso nos agentes
def analyze_data_for_ml(df: pd.DataFrame) -> str:
    """Função para análise de dados - uso em agentes"""
    try:
        preprocessor = DataPreprocessor()
        analysis = preprocessor.analyze_data_quality(df)
        return json.dumps(analysis, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"Erro na análise: {str(e)}"


def train_ml_models(df: pd.DataFrame, target_column: str, test_size: float = 0.2) -> str:
    """Função para treinamento de modelos - uso em agentes"""
    try:
        if not ML_AVAILABLE:
            return "Bibliotecas de Machine Learning não estão disponíveis. Instale: pip install scikit-learn xgboost imbalanced-learn"
        
        # Pré-processamento
        preprocessor = DataPreprocessor()
        data_prep = preprocessor.prepare_data_for_ml(df, target_column, test_size)
        
        # Treinamento
        trainer = ModelTrainer()
        results = trainer.train_multiple_models(
            data_prep["X_train"],
            data_prep["y_train"], 
            data_prep["X_test"],
            data_prep["y_test"],
            data_prep["is_classification"]
        )
        
        # Avaliação
        evaluator = ModelEvaluator()
        summary = evaluator.generate_model_summary(results, data_prep["is_classification"])
        
        # Formatação do resultado
        output = {
            "data_info": {
                "problem_type": "Classificação" if data_prep["is_classification"] else "Regressão",
                "features": len(data_prep["feature_names"]),
                "training_samples": data_prep["original_shapes"]["X_train"][0],
                "test_samples": data_prep["original_shapes"]["X_test"][0]
            },
            "model_summary": summary,
            "detailed_results": {
                model: {
                    "metrics": result["metrics"] if "metrics" in result else {"error": result.get("error", "Unknown error")}
                }
                for model, result in results.items()
            }
        }
        
        return json.dumps(output, ensure_ascii=False, indent=2)
        
    except Exception as e:
        return f"Erro no treinamento: {str(e)}"


# Função para criar gráficos que funciona no Streamlit
def create_ml_comparison_plots(df: pd.DataFrame, target_column: str):
    """Cria gráficos de comparação para Streamlit"""
    try:
        if not ML_AVAILABLE:
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, "Bibliotecas de ML não disponíveis", 
                    ha='center', va='center', fontsize=16)
            plt.axis('off')
            return plt.gcf()
        
        # Processar dados
        preprocessor = DataPreprocessor()
        data_prep = preprocessor.prepare_data_for_ml(df, target_column)
        
        # Treinar modelos
        trainer = ModelTrainer()
        results = trainer.train_multiple_models(
            data_prep["X_train"],
            data_prep["y_train"], 
            data_prep["X_test"],
            data_prep["y_test"],
            data_prep["is_classification"]
        )
        
        # Criar gráficos
        evaluator = ModelEvaluator()
        fig1, fig2 = evaluator.create_comparison_plots(
            results, data_prep["y_test"], data_prep["is_classification"]
        )
        
        return fig1  # Retorna o primeiro gráfico
        
    except Exception as e:
        plt.figure(figsize=(10, 6))
        plt.text(0.5, 0.5, f"Erro: {str(e)}", ha='center', va='center', fontsize=12)
        plt.axis('off')
        return plt.gcf()