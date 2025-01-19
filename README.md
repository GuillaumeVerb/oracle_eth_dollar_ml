# Hybrid Oracle ETH/USD and S&P 500 Index Project

This project implements a hybrid oracle system that combines ETH/USD price and S&P 500 index data. It consists of Python scripts for data collection and model training, and a Node.js script for automated oracle updates.

## Project Components

1. **Python Scripts**
   - `train_model.py`: Machine learning model training
   - `model_validation.py`: Advanced model validation and diagnostics
   - `predict_and_monitor.py`: Real-time price monitoring and predictions

2. **Node.js Script**
   - `update_oracle.js`: Smart contract oracle updater

## Python Components

### 1. Model Training (`train_model.py`)

#### Features
- Hybrid model combining ETH and S&P 500 data
- Advanced feature engineering
- Multiple model comparison (XGBoost, LightGBM)
- Hyperparameter optimization
- Cross-validation with time series split

#### Key Components
```python
class HybridIndexPredictor:
    def __init__(self):
        self.models = {
            'xgboost': XGBRegressor(),
            'lightgbm': LGBMRegressor()
        }
```

#### Technical Features
- Volatility calculations
- Moving averages (7, 14, 30 days)
- Momentum indicators
- Price ratios and correlations

#### Model Evaluation
- R-squared score
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)

### 2. Model Validation (`model_validation.py`)

#### Validation Features
```python
class ModelValidator:
    def check_residuals(self)
    def temporal_stability(self)
    def feature_importance_analysis()
```

#### Diagnostic Tools
- Residual analysis
- Heteroscedasticity tests
- Autocorrelation checks
- SHAP value analysis

#### Visualization
- Feature importance plots
- Residual distribution
- Temporal stability graphs
- SHAP summary plots

### 3. Price Monitoring (`predict_and_monitor.py`)

#### Features
- Real-time ETH price monitoring via CoinGecko
- S&P 500 data from Alpha Vantage
- Automated alerts system
- Historical data tracking

#### Configuration
```python
# Environment variables
COINGECKO_API_URL=
ALPHAVANTAGE_API_KEY=
ALERT_THRESHOLD=0.05  # 5% threshold
```

## Node.js Oracle Updater

### Overview

The Node.js script automates updates to the HybridOracle smart contract by monitoring ETH/USD and S&P 500 index values.

### Architecture

#### Main Components

1. **OracleUpdater Class**
   - Update logic management
   - Smart contract interaction
   - Data validation

2. **Configuration System**
   - `.env` for sensitive variables
   - Configurable update thresholds
   - Retry parameters

3. **Logging System**
   - Updates logged to `oracle_updates.json`
   - Detailed console logs
   - Transaction tracking

### Detailed Features

#### 1. Data Reading
```javascript
async getLatestDataFromCSV()
```
- Reads `historical_data.csv`
- Parses ETH and S&P 500 values
- Data validation

#### 2. State Verification
```javascript
async checkContractState()
```
- Contract pause check
- Wallet permissions
- Operation security

#### 3. Update Validation
```javascript
async checkUpdateNeeded(latestData)
```
- On-chain data comparison
- Variation threshold (5%)
- External alerts via `alert.json`

### Security

#### Protection Measures
1. **Data Validation**
   - Null value checks
   - Format validation
   - Boundary controls

2. **Transaction Management**
   - Gas buffer (20%)
   - Confirmation waiting
   - Event verification

## Installation and Setup

### Prerequisites
- Python 3.8+
- Node.js v14+
- npm or yarn
- Ethereum node access

### Python Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Unix
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Node.js Setup
```bash
# Install dependencies
npm install

# Configure environment
cp .env.example .env
# Edit .env with your values
```

## Usage

### Training the Model
```bash
python train_model.py
```

### Validating the Model
```bash
python model_validation.py
```

### Running the Price Monitor
```bash
python predict_and_monitor.py
```

### Starting the Oracle Updater
```bash
npm start
# or
node update_oracle.js
```

## Configuration

### Environment Variables
```env
# API Configuration
WEB3_PROVIDER_URL=https://mainnet.infura.io/v3/your_key
WALLET_PRIVATE_KEY=your_private_key
CONTRACT_ADDRESS=your_contract_address
COINGECKO_API_URL=https://api.coingecko.com/api/v3
ALPHAVANTAGE_API_KEY=your_key

# Update Parameters
UPDATE_THRESHOLD=0.05
MAX_RETRIES=3
RETRY_DELAY=5000
```

## Error Handling

### Types of Errors
1. **Reading Errors**
   - Missing CSV file
   - Invalid data format

2. **Blockchain Errors**
   - Failed transactions
   - Insufficient gas
   - Incorrect nonce

3. **Validation Errors**
   - Out-of-bounds data
   - Contract paused
   - Insufficient permissions

### Retry System
- Multiple transaction attempts
- Exponential delays
- Detailed error logs

## Maintenance

### Regular Tasks
1. Check error logs
2. Monitor gas costs
3. Validate update thresholds
4. Review model performance

### Recommended Updates
1. Adjust thresholds based on market
2. Optimize gas parameters
3. Update dependencies
4. Retrain model periodically

## Support and Contribution

### Contributing
1. Fork the repository
2. Create a feature branch
3. Submit a Pull Request

### Support
- Open issues for bugs
- Consult smart contract documentation
- Check logs for debugging

## License
MIT License 