require('dotenv').config();
const ethers = require('ethers');
const fs = require('fs');
const csv = require('csv-parser');
const path = require('path');

class OracleUpdater {
    constructor() {
        // Configuration du provider et du wallet
        this.provider = new ethers.providers.JsonRpcProvider(process.env.WEB3_PROVIDER_URL);
        this.wallet = new ethers.Wallet(process.env.WALLET_PRIVATE_KEY, this.provider);
        
        // Configuration du contrat
        this.contractAddress = process.env.CONTRACT_ADDRESS;
        this.contractABI = require('./contracts/HybridOracle.json').abi;
        this.contract = new ethers.Contract(
            this.contractAddress,
            this.contractABI,
            this.wallet
        );

        // Configuration des seuils
        this.UPDATE_THRESHOLD = 0.05; // 5% de différence
        this.MAX_RETRIES = 3;
        this.RETRY_DELAY = 5000; // 5 secondes
    }

    async getLatestDataFromCSV() {
        return new Promise((resolve, reject) => {
            const results = [];
            fs.createReadStream('historical_data.csv')
                .pipe(csv())
                .on('data', (data) => results.push(data))
                .on('end', () => {
                    if (results.length === 0) {
                        reject(new Error('No data found in CSV file'));
                        return;
                    }
                    // Prendre la dernière ligne
                    const latestData = results[results.length - 1];
                    resolve({
                        ethPrice: parseFloat(latestData.eth_price),
                        sp500Index: parseFloat(latestData.sp500_index),
                        timestamp: new Date(latestData.timestamp)
                    });
                })
                .on('error', (error) => reject(error));
        });
    }

    async checkContractState() {
        try {
            const isPaused = await this.contract.paused();
            if (isPaused) {
                throw new Error('Contract is paused');
            }

            const owner = await this.contract.owner();
            if (owner.toLowerCase() !== this.wallet.address.toLowerCase()) {
                throw new Error('Wallet is not the contract owner');
            }

            return true;
        } catch (error) {
            console.error('Contract state check failed:', error);
            throw error;
        }
    }

    async checkUpdateNeeded(latestData) {
        try {
            // Vérifier si une alerte existe
            const alertFile = path.join(__dirname, 'alert.json');
            if (fs.existsSync(alertFile)) {
                const alert = JSON.parse(fs.readFileSync(alertFile));
                if (alert.timestamp > Date.now() - 3600000) { // Alert moins vieille qu'une heure
                    return true;
                }
            }

            // Vérifier les données on-chain actuelles
            const currentEthPrice = await this.contract.ethPrice();
            const currentSp500Index = await this.contract.sp500Index();
            
            // Calculer les différences
            const ethPriceDiff = Math.abs(
                (latestData.ethPrice * 100 - currentEthPrice) / currentEthPrice
            );
            const sp500Diff = Math.abs(
                (latestData.sp500Index - currentSp500Index) / currentSp500Index
            );

            return ethPriceDiff > this.UPDATE_THRESHOLD || sp500Diff > this.UPDATE_THRESHOLD;
        } catch (error) {
            console.error('Update check failed:', error);
            throw error;
        }
    }

    async updateOracle(ethPrice, sp500Index, retryCount = 0) {
        try {
            console.log(`Updating oracle with ETH price: ${ethPrice}, S&P 500: ${sp500Index}`);

            // Convertir le prix ETH en centimes
            const ethPriceScaled = Math.round(ethPrice * 100);
            const sp500IndexScaled = Math.round(sp500Index);

            // Estimer le gas
            const gasEstimate = await this.contract.estimateGas.updateData(
                ethPriceScaled,
                sp500IndexScaled
            );

            // Préparer la transaction avec un buffer de gas
            const tx = await this.contract.updateData(
                ethPriceScaled,
                sp500IndexScaled,
                {
                    gasLimit: Math.round(gasEstimate * 1.2) // 20% buffer
                }
            );

            console.log(`Transaction sent: ${tx.hash}`);
            
            // Attendre la confirmation
            const receipt = await tx.wait();
            console.log(`Transaction confirmed in block ${receipt.blockNumber}`);

            // Vérifier l'événement DataUpdated
            const event = receipt.events.find(e => e.event === 'DataUpdated');
            if (event) {
                console.log('Data successfully updated on-chain');
                this.logUpdate({
                    timestamp: new Date(),
                    txHash: tx.hash,
                    ethPrice,
                    sp500Index,
                    blockNumber: receipt.blockNumber
                });
            }

            return receipt;
        } catch (error) {
            if (retryCount < this.MAX_RETRIES) {
                console.log(`Retry attempt ${retryCount + 1}/${this.MAX_RETRIES}`);
                await new Promise(resolve => setTimeout(resolve, this.RETRY_DELAY));
                return this.updateOracle(ethPrice, sp500Index, retryCount + 1);
            }
            console.error('Oracle update failed:', error);
            throw error;
        }
    }

    logUpdate(updateData) {
        const logFile = 'oracle_updates.json';
        let logs = [];
        
        if (fs.existsSync(logFile)) {
            logs = JSON.parse(fs.readFileSync(logFile));
        }
        
        logs.push(updateData);
        fs.writeFileSync(logFile, JSON.stringify(logs, null, 2));
    }
}

async function main() {
    const updater = new OracleUpdater();
    
    try {
        // Vérifier l'état du contrat
        await updater.checkContractState();
        
        // Obtenir les dernières données
        const latestData = await updater.getLatestDataFromCSV();
        console.log('Latest data:', latestData);
        
        // Vérifier si une mise à jour est nécessaire
        const needsUpdate = await updater.checkUpdateNeeded(latestData);
        
        if (needsUpdate) {
            console.log('Update needed, proceeding with oracle update...');
            const receipt = await updater.updateOracle(
                latestData.ethPrice,
                latestData.sp500Index
            );
            console.log('Update successful!');
        } else {
            console.log('No update needed at this time');
        }
    } catch (error) {
        console.error('Error in main process:', error);
        process.exit(1);
    }
}

// Exécuter le script
if (require.main === module) {
    main().catch(console.error);
}

module.exports = OracleUpdater; 