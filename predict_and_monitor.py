import os
import requests
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import time
from dotenv import load_dotenv
import logging
from web3 import Web3
import json
import asyncio

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring.log'),
        logging.StreamHandler()
    ]
)

class HybridIndexMonitor:
    def __init__(self, model_path='hybrid_model.pkl', history_path='prediction_history.csv'):
        """
        Initialise le moniteur d'index hybride
        """
        # Charger les variables d'environnement
        load_dotenv()
        
        # Configuration
        self.coingecko_url = os.getenv('COINGECKO_API_URL')
        self.alphavantage_url = os.getenv('ALPHAVANTAGE_API_URL')
        self.alphavantage_key = os.getenv('ALPHAVANTAGE_API_KEY')
        self.threshold = float(os.getenv('ALERT_THRESHOLD', '0.05'))  # 5% par défaut
        self.history_path = history_path
        
        # Smart contract configuration
        self.web3 = Web3(Web3.HTTPProvider(os.getenv('WEB3_PROVIDER_URL')))
        self.contract_address = os.getenv('CONTRACT_ADDRESS')
        self.contract_abi = json.loads(os.getenv('CONTRACT_ABI'))
        self.contract = self.web3.eth.contract(
            address=self.contract_address,
            abi=self.contract_abi
        )
        self.wallet_private_key = os.getenv('WALLET_PRIVATE_KEY')
        
        # Charger le modèle
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            logging.info("Modèle chargé avec succès")
        except Exception as e:
            logging.error(f"Erreur lors du chargement du modèle: {str(e)}")
            raise
        
        # Initialiser l'historique
        self.initialize_history()
    
    def initialize_history(self):
        """
        Initialise ou charge l'historique des prédictions
        """
        if os.path.exists(self.history_path):
            self.history = pd.read_csv(self.history_path)
        else:
            self.history = pd.DataFrame(columns=[
                'timestamp', 'eth_price', 'sp500_index', 'actual_hybrid_index',
                'predicted_hybrid_index', 'difference_percentage', 'alert_triggered'
            ])
    
    def get_eth_price(self):
        """
        Récupère le prix ETH via CoinGecko
        """
        try:
            response = requests.get(self.coingecko_url)
            response.raise_for_status()
            data = response.json()
            return float(data['ethereum']['usd'])
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du prix ETH: {str(e)}")
            raise
    
    def get_sp500_value(self):
        """
        Récupère la valeur du S&P 500 via Alpha Vantage
        """
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': 'SPX',
                'apikey': self.alphavantage_key
            }
            response = requests.get(self.alphavantage_url, params=params)
            response.raise_for_status()
            data = response.json()
            return float(data['Global Quote']['05. price'])
        except Exception as e:
            logging.error(f"Erreur lors de la récupération du S&P 500: {str(e)}")
            raise
    
    def predict_hybrid_index(self, eth_price, sp500_index):
        """
        Prédit l'index hybride en utilisant le modèle
        """
        try:
            # Préparer les features comme attendu par le modèle
            features = pd.DataFrame({
                'eth_price': [eth_price],
                'sp500_index': [sp500_index]
            })
            
            # Faire la prédiction
            prediction = self.model.predict(features)[0]
            return prediction
        except Exception as e:
            logging.error(f"Erreur lors de la prédiction: {str(e)}")
            raise
    
    def calculate_actual_hybrid_index(self, eth_price, sp500_index):
        """
        Calcule l'index hybride actuel
        """
        return (eth_price / sp500_index) * 1000
    
    def update_smart_contract(self, eth_price, sp500_index):
        """
        Met à jour le smart contract avec les nouvelles valeurs
        """
        try:
            # Préparer la transaction
            nonce = self.web3.eth.get_transaction_count(self.web3.eth.account.from_key(self.wallet_private_key).address)
            
            # Construire la transaction
            transaction = self.contract.functions.updateData(
                int(eth_price * 100),  # Convertir en centimes
                int(sp500_index)
            ).build_transaction({
                'gas': 200000,
                'gasPrice': self.web3.eth.gas_price,
                'nonce': nonce,
            })
            
            # Signer et envoyer la transaction
            signed_txn = self.web3.eth.account.sign_transaction(
                transaction,
                self.wallet_private_key
            )
            tx_hash = self.web3.eth.send_raw_transaction(signed_txn.rawTransaction)
            
            # Attendre la confirmation
            receipt = self.web3.eth.wait_for_transaction_receipt(tx_hash)
            logging.info(f"Smart contract mis à jour. Transaction hash: {receipt['transactionHash'].hex()}")
            
        except Exception as e:
            logging.error(f"Erreur lors de la mise à jour du smart contract: {str(e)}")
            raise
    
    def save_to_history(self, data):
        """
        Sauvegarde les données dans l'historique
        """
        self.history = pd.concat([self.history, pd.DataFrame([data])], ignore_index=True)
        self.history.to_csv(self.history_path, index=False)
    
    def monitor_and_predict(self):
        """
        Fonction principale de monitoring et prédiction
        """
        try:
            # Récupérer les données actuelles
            eth_price = self.get_eth_price()
            sp500_index = self.get_sp500_value()
            
            # Calculer l'index hybride actuel
            actual_hybrid_index = self.calculate_actual_hybrid_index(eth_price, sp500_index)
            
            # Prédire l'index hybride
            predicted_hybrid_index = self.predict_hybrid_index(eth_price, sp500_index)
            
            # Calculer la différence en pourcentage
            difference_percentage = abs(
                (predicted_hybrid_index - actual_hybrid_index) / actual_hybrid_index
            )
            
            # Vérifier si une alerte est nécessaire
            alert_triggered = difference_percentage > self.threshold
            
            # Préparer les données pour l'historique
            data = {
                'timestamp': datetime.now().isoformat(),
                'eth_price': eth_price,
                'sp500_index': sp500_index,
                'actual_hybrid_index': actual_hybrid_index,
                'predicted_hybrid_index': predicted_hybrid_index,
                'difference_percentage': difference_percentage,
                'alert_triggered': alert_triggered
            }
            
            # Sauvegarder dans l'historique
            self.save_to_history(data)
            
            # Logger les résultats
            logging.info(f"Prédiction effectuée: Actuel={actual_hybrid_index:.2f}, "
                        f"Prédit={predicted_hybrid_index:.2f}, "
                        f"Différence={difference_percentage*100:.2f}%")
            
            # Si la différence est importante, mettre à jour le smart contract
            if alert_triggered:
                logging.warning(f"Alerte! Différence de {difference_percentage*100:.2f}% "
                              f"détectée. Mise à jour du smart contract...")
                self.update_smart_contract(eth_price, sp500_index)
            
            return data
            
        except Exception as e:
            logging.error(f"Erreur dans le cycle de monitoring: {str(e)}")
            raise

def main():
    # Configuration
    MONITORING_INTERVAL = int(os.getenv('MONITORING_INTERVAL', '300'))  # 5 minutes par défaut
    
    monitor = HybridIndexMonitor()
    logging.info("Démarrage du monitoring de l'index hybride")
    
    while True:
        try:
            monitor.monitor_and_predict()
            time.sleep(MONITORING_INTERVAL)
            
        except KeyboardInterrupt:
            logging.info("Arrêt du monitoring")
            break
            
        except Exception as e:
            logging.error(f"Erreur critique: {str(e)}")
            time.sleep(60)  # Attendre 1 minute avant de réessayer

if __name__ == "__main__":
    main() 