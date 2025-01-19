from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer
import optuna
import shap
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats

class ModelValidator:
    def __init__(self, model, X, y):
        self.model = model
        self.X = X
        self.y = y
        
    def check_residuals(self, y_true, y_pred):
        """
        Analyse complète des résidus
        """
        residuals = y_true - y_pred
        
        # Tests statistiques
        normality = stats.shapiro(residuals)
        heteroscedasticity = stats.spearmanr(y_pred, np.abs(residuals))
        autocorrelation = acorr_ljungbox(residuals, lags=10)
        
        # Visualisations
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # QQ Plot
        stats.probplot(residuals, dist="norm", plot=axes[0, 0])
        axes[0, 0].set_title("Q-Q Plot")
        
        # Distribution des résidus
        sns.histplot(residuals, kde=True, ax=axes[0, 1])
        axes[0, 1].set_title("Distribution des résidus")
        
        # Résidus vs Prédictions
        axes[1, 0].scatter(y_pred, residuals, alpha=0.5)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_title("Résidus vs Prédictions")
        
        # Autocorrélation
        pd.plotting.autocorrelation_plot(residuals, ax=axes[1, 1])
        
        plt.tight_layout()
        plt.savefig('residuals_analysis.png')
        plt.close()
        
        return {
            'normality_test': normality,
            'heteroscedasticity_test': heteroscedasticity,
            'autocorrelation_test': autocorrelation
        }
    
    def temporal_stability(self, X_train, X_test, y_train, y_test):
        """
        Analyse de la stabilité temporelle du modèle
        """
        # Performance sur des fenêtres glissantes
        tscv = TimeSeriesSplit(n_splits=5)
        window_scores = []
        
        for train_idx, val_idx in tscv.split(X_train):
            X_window_train = X_train.iloc[train_idx]
            X_window_val = X_train.iloc[val_idx]
            y_window_train = y_train.iloc[train_idx]
            y_window_val = y_train.iloc[val_idx]
            
            self.model.fit(X_window_train, y_window_train)
            score = self.model.score(X_window_val, y_window_val)
            window_scores.append(score)
        
        # Plot de la stabilité temporelle
        plt.figure(figsize=(10, 6))
        plt.plot(window_scores, marker='o')
        plt.title('Stabilité temporelle du modèle')
        plt.xlabel('Fenêtre temporelle')
        plt.ylabel('Score R²')
        plt.savefig('temporal_stability.png')
        plt.close()
        
        return window_scores
    
    def feature_importance_analysis(self):
        """
        Analyse approfondie de l'importance des features avec SHAP
        """
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer.shap_values(self.X)
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, self.X, show=False)
        plt.savefig('shap_summary.png')
        plt.close()
        
        # Feature interactions
        plt.figure(figsize=(12, 8))
        shap.dependence_plot(
            "eth_price", shap_values, self.X,
            interaction_index="sp500_index",
            show=False
        )
        plt.savefig('feature_interactions.png')
        plt.close()
        
        return shap_values 