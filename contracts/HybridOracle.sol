// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract HybridOracle {
    address public owner;
    uint256 public ethPrice;      // ETH/USD price * 100
    uint256 public sp500Index;    // S&P 500 index value
    uint256 public hybridIndex;   // Calculated hybrid index
    uint256 public lastUpdate;    // Last update timestamp

    event DataUpdated(
        uint256 ethPrice,
        uint256 sp500Index,
        uint256 hybridIndex,
        uint256 timestamp
    );

    event OwnershipTransferred(
        address indexed previousOwner,
        address indexed newOwner
    );

    modifier onlyOwner() {
        require(msg.sender == owner, "HybridOracle: caller is not the owner");
        _;
    }

    constructor() {
        owner = msg.sender;
    }

    function updateData(uint256 _ethPrice, uint256 _sp500Index) external onlyOwner {
        require(_sp500Index > 0, "HybridOracle: S&P 500 index must be greater than 0");
        
        ethPrice = _ethPrice;
        sp500Index = _sp500Index;
        
        // Calculate hybrid index: (ethPrice * 1000) / sp500Index
        hybridIndex = (ethPrice * 1000) / sp500Index;
        lastUpdate = block.timestamp;

        emit DataUpdated(ethPrice, sp500Index, hybridIndex, lastUpdate);
    }

    function transferOwnership(address newOwner) external onlyOwner {
        require(newOwner != address(0), "HybridOracle: new owner is the zero address");
        
        address oldOwner = owner;
        owner = newOwner;
        
        emit OwnershipTransferred(oldOwner, newOwner);
    }
} 