# HybridOracle Smart Contract

## Overview

HybridOracle is a secure Ethereum smart contract designed to maintain and provide on-chain price data for ETH/USD and the S&P 500 index, while calculating a hybrid index based on these values. The contract implements multiple security features and follows best practices for financial oracle implementations.

## Features

### Core Functionality
- Stores ETH/USD price (scaled by 100 for 2 decimal precision)
- Stores S&P 500 index value
- Calculates a hybrid index using high-precision mathematics
- Provides data freshness validation

### Security Features
- Owner-based access control
- Two-step ownership transfer with time delay
- Price change rate limiting
- Emergency pause mechanism
- Protection against flash loan attacks
- Reentrancy protection
- ETH recovery for accidental transfers

## Contract Architecture

### State Variables
```solidity
address public owner;                // Contract owner address
address public pendingOwner;        // Address of pending owner during transfer
uint256 public ethPrice;            // ETH/USD price * 100
uint256 public sp500Index;          // S&P 500 index value
uint256 public hybridIndex;         // Calculated hybrid index
uint256 public lastUpdate;          // Last update timestamp
bool public paused;                 // Circuit breaker
```

### Constants
```solidity
MAX_ETH_PRICE = 1000000 * 100      // Maximum $1M ETH price
MAX_SP500_VALUE = 100000           // Maximum 100,000 S&P 500 value
OWNERSHIP_TRANSFER_DELAY = 2 days   // Ownership transfer delay
MAX_PRICE_CHANGE_PERCENTAGE = 20    // 20% max price change
MAX_UPDATE_DELAY = 24 hours        // Maximum time between updates
```

## Security Measures

### Price Protection
- Maximum bounds for both ETH price and S&P 500 index
- Rate limiting: maximum 20% price change between updates
- High precision calculations to prevent rounding errors
- Freshness checks to prevent use of stale data

### Access Control
- Owner-only functions for critical operations
- Two-step ownership transfer with 48-hour delay
- Ability to cancel pending ownership transfers
- Emergency pause functionality

### Contract Safety
- Reentrancy guard on price updates
- ETH recovery mechanism
- Custom errors for gas optimization
- Comprehensive event emission

## Functions

### Core Functions

#### `updateData`
Updates price data with multiple safety checks:
```solidity
function updateData(uint256 _ethPrice, uint256 _sp500Index) external
```
- Requires owner access
- Validates price ranges
- Checks price change rates
- Updates hybrid index
- Emits events

#### `getLatestData`
Retrieves current price data:
```solidity
function getLatestData() external view returns (
    uint256 _ethPrice,
    uint256 _sp500Index,
    uint256 _hybridIndex,
    uint256 _lastUpdate
)
```
- Checks data freshness
- Validates contract not paused

### Administrative Functions

#### Ownership Management
```solidity
function initiateOwnershipTransfer(address newOwner) external
function completeOwnershipTransfer() external
function cancelOwnershipTransfer() external
```

#### Emergency Controls
```solidity
function pause() external
function unpause() external
function emergencyShutdown(string calldata reason) external
```

## Events and Errors

### Events
```solidity
event DataUpdated(uint256 ethPrice, uint256 sp500Index, uint256 hybridIndex, uint256 timestamp);
event OwnershipTransferInitiated(address indexed previousOwner, address indexed newOwner);
event OwnershipTransferred(address indexed previousOwner, address indexed newOwner);
event Paused(address account);
event Unpaused(address account);
event EthReceived(address sender, uint256 amount);
event EthRecovered(address recipient, uint256 amount);
event OwnershipTransferCancelled(address owner, address canceledOwner);
```

### Custom Errors
```solidity
error InvalidPrice(uint256 price);
error InvalidIndex(uint256 index);
error PriceChangeTooBig(uint256 oldPrice, uint256 newPrice);
error UnauthorizedCaller(address caller);
error NoTransferPending();
error TransferDelayNotMet(uint256 initiatedTime, uint256 requiredDelay);
error EmergencyShutdown(string reason);
error StaleData(uint256 lastUpdateTime, uint256 maxDelay);
```

## Usage Guidelines

### Best Practices
1. Monitor emitted events for tracking updates
2. Implement off-chain alerting for price changes
3. Regular validation of data freshness
4. Test emergency procedures periodically
5. Maintain secure owner key management

### Security Considerations
1. Owner privileges require careful management
2. Price updates should be monitored for manipulation
3. Emergency procedures should be tested regularly
4. Respect time delays for ownership transfers

## Development and Testing

### Requirements
- Solidity ^0.8.0
- Ethereum development environment (Hardhat/Truffle)
- OpenZeppelin Contracts (recommended for testing)

### Testing Recommendations
1. Unit tests for all core functions
2. Price change boundary tests
3. Ownership transfer scenarios
4. Emergency procedure testing
5. Gas optimization validation

## License
MIT License

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request. 