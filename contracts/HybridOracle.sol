// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract HybridOracle {
    address public owner;
    address public pendingOwner;
    uint256 public ethPrice;      // ETH/USD price * 100
    uint256 public sp500Index;    // S&P 500 index value
    uint256 public hybridIndex;   // Calculated hybrid index
    uint256 public lastUpdate;    // Last update timestamp
    bool public paused;
    
    uint256 private constant MAX_ETH_PRICE = 1000000 * 100; // $1M max price
    uint256 private constant MAX_SP500_VALUE = 100000;      // 100,000 max value
    uint256 private constant OWNERSHIP_TRANSFER_DELAY = 2 days;
    uint256 private constant MAX_PRICE_CHANGE_PERCENTAGE = 20; // 20% max change
    uint256 private constant MAX_UPDATE_DELAY = 24 hours;      // Maximum time between updates
    uint256 private ownershipTransferInitiated;

    event DataUpdated(
        uint256 ethPrice,
        uint256 sp500Index,
        uint256 hybridIndex,
        uint256 timestamp
    );

    event OwnershipTransferInitiated(
        address indexed previousOwner,
        address indexed newOwner
    );

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    event Paused(address account);
    event Unpaused(address account);

    error EmergencyShutdown(string reason);
    error StaleData(uint256 lastUpdateTime, uint256 maxDelay);
    error InvalidPrice(uint256 price);
    error InvalidIndex(uint256 index);
    error PriceChangeTooBig(uint256 oldPrice, uint256 newPrice);
    error UnauthorizedCaller(address caller);
    error NoTransferPending();
    error TransferDelayNotMet(uint256 initiatedTime, uint256 requiredDelay);

    modifier onlyOwner() {
        require(msg.sender == owner, "HybridOracle: caller is not the owner");
        _;
    }

    modifier whenNotPaused() {
        require(!paused, "HybridOracle: contract is paused");
        _;
    }

    modifier nonReentrant() {
        require(msg.sender != tx.origin || msg.sender == owner, "HybridOracle: no contract calls");
        _;
    }

    modifier freshData() {
        if (lastUpdate + MAX_UPDATE_DELAY < block.timestamp) {
            revert StaleData(lastUpdate, MAX_UPDATE_DELAY);
        }
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function updateData(uint256 _ethPrice, uint256 _sp500Index) 
        external 
        onlyOwner 
        whenNotPaused 
        nonReentrant 
    {
        if (_sp500Index == 0) revert InvalidIndex(0);
        if (_ethPrice == 0) revert InvalidPrice(0);
        if (_ethPrice > MAX_ETH_PRICE) revert InvalidPrice(_ethPrice);
        if (_sp500Index > MAX_SP500_VALUE) revert InvalidIndex(_sp500Index);
        
        // Check for suspicious price changes with more precise calculations
        if (ethPrice != 0) {
            uint256 priceChangeNumerator = _ethPrice > ethPrice 
                ? (_ethPrice - ethPrice) * 10000
                : (ethPrice - _ethPrice) * 10000;
            
            if ((priceChangeNumerator / ethPrice) > (MAX_PRICE_CHANGE_PERCENTAGE * 100)) {
                revert PriceChangeTooBig(ethPrice, _ethPrice);
            }
        }
        
        if (sp500Index != 0) {
            uint256 sp500Change = _sp500Index > sp500Index 
                ? ((_sp500Index - sp500Index) * 100) / sp500Index 
                : ((sp500Index - _sp500Index) * 100) / sp500Index;
            require(
                sp500Change <= MAX_PRICE_CHANGE_PERCENTAGE,
                "HybridOracle: S&P 500 change too high"
            );
        }

        ethPrice = _ethPrice;
        sp500Index = _sp500Index;
        
        // Calculate hybrid index: (ethPrice * 1000) / sp500Index
        hybridIndex = calculateHybridIndex(_ethPrice, _sp500Index);
        lastUpdate = uint256(block.timestamp);

        emit DataUpdated(ethPrice, sp500Index, hybridIndex, lastUpdate);
    }

    function initiateOwnershipTransfer(address newOwner) external onlyOwner {
        require(newOwner != address(0), "HybridOracle: new owner is the zero address");
        require(newOwner != owner, "HybridOracle: new owner is the current owner");
        
        pendingOwner = newOwner;
        ownershipTransferInitiated = block.timestamp;
        
        emit OwnershipTransferInitiated(owner, newOwner);
    }

    function completeOwnershipTransfer() external {
        require(msg.sender == pendingOwner, "HybridOracle: caller is not the pending owner");
        require(pendingOwner != address(0), "HybridOracle: no pending ownership transfer");
        require(
            block.timestamp >= ownershipTransferInitiated + OWNERSHIP_TRANSFER_DELAY,
            "HybridOracle: transfer delay not met"
        );

        address oldOwner = owner;
        owner = pendingOwner;
        pendingOwner = address(0);
        ownershipTransferInitiated = 0;
        
        emit OwnershipTransferred(oldOwner, owner);
    }

    function pause() external onlyOwner {
        paused = true;
        emit Paused(msg.sender);
    }

    function unpause() external onlyOwner {
        paused = false;
        emit Unpaused(msg.sender);
    }

    function getLatestData() external view 
        whenNotPaused 
        freshData 
        returns (
        uint256 _ethPrice,
        uint256 _sp500Index,
        uint256 _hybridIndex,
        uint256 _lastUpdate
    ) {
        return (ethPrice, sp500Index, hybridIndex, lastUpdate);
    }

    function emergencyShutdown(string calldata reason) external onlyOwner {
        paused = true;
        emit Paused(msg.sender);
        revert EmergencyShutdown(reason);
    }

    function validateTimestamp(uint256 timestamp) internal view {
        require(
            timestamp >= block.timestamp - 1 hours &&
            timestamp <= block.timestamp + 1 hours,
            "HybridOracle: invalid timestamp"
        );
    }

    // Receive function to accept ETH sent by mistake
    receive() external payable {
        emit EthReceived(msg.sender, msg.value);
    }

    // Function to recover ETH sent by mistake
    function recoverEth() external onlyOwner {
        uint256 balance = address(this).balance;
        require(balance > 0, "HybridOracle: no ETH to recover");
        
        (bool success, ) = owner.call{value: balance}("");
        require(success, "HybridOracle: ETH transfer failed");
        
        emit EthRecovered(owner, balance);
    }

    function cancelOwnershipTransfer() external onlyOwner {
        if (pendingOwner == address(0)) revert NoTransferPending();
        
        address canceledOwner = pendingOwner;
        pendingOwner = address(0);
        ownershipTransferInitiated = 0;
        
        emit OwnershipTransferCancelled(owner, canceledOwner);
    }

    // Improved precision for hybrid index calculation
    function calculateHybridIndex(uint256 _ethPrice, uint256 _sp500Index) 
        internal 
        pure 
        returns (uint256) 
    {
        // Use higher precision for calculation
        uint256 scaleFactor = 1e18;
        uint256 scaledEthPrice = _ethPrice * scaleFactor;
        return (scaledEthPrice * 1000) / (_sp500Index * 1e3);
    }
} 