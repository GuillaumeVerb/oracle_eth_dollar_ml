import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Générer des dates
start_date = datetime(2023, 1, 1)
dates = [start_date + timedelta(days=x) for x in range(365)]

# Générer des prix ETH simulés (commençant à ~$2000 avec de la volatilité)
np.random.seed(42)
eth_price = 2000 + np.random.normal(0, 100, len(dates)).cumsum()
eth_price = np.maximum(eth_price, 1000)  # Prix minimum de 1000

# Générer des valeurs S&P 500 simulées (commençant à ~4000 avec moins de volatilité)
sp500_index = 4000 + np.random.normal(0, 20, len(dates)).cumsum()
sp500_index = np.maximum(sp500_index, 3500)  # Valeur minimum de 3500

# Calculer l'indice hybride
hybrid_index = (eth_price / sp500_index) * 1000

# Ajouter du bruit pour simuler les prédictions
predicted_hybrid_index = hybrid_index + np.random.normal(0, hybrid_index * 0.05, len(dates))

# Créer le DataFrame
df = pd.DataFrame({
    'timestamp': dates,
    'eth_price': eth_price,
    'sp500_index': sp500_index,
    'hybrid_index': hybrid_index,
    'predicted_hybrid_index': predicted_hybrid_index
})

# Sauvegarder en CSV
df.to_csv('historical_data.csv', index=False)
print("Données exemple générées dans 'historical_data.csv'") 