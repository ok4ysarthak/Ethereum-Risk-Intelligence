// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title FraudDetection System
 * @dev Optimized for gas efficiency and real-world batch processing.
 */
contract FraudDetection {
    
    // --- Structs ---
    struct TransactionRecord {
        uint256 value;      // Stored in wei
        uint48 timestamp;   // Optimized for storage (valid until year 8,000,000)
        uint8 riskScore;    // 0-100 scale (uint8 saves gas)
    }
    
    struct WalletInfo {
        uint8 trustScore;   // 0-100 scale
        uint48 lastUpdated; // Optimized timestamp
    }

    // --- State Variables ---
    address public owner;
    
    // Mappings
    mapping(bytes32 => TransactionRecord) private transactions;
    mapping(address => WalletInfo) private wallets;
    mapping(address => bool) public authorizedOracles;

    // Configuration
    uint8 public minTrustScore = 30; 
    uint8 public highRiskThreshold = 70; 

    // --- Events (Crucial for your Frontend) ---
    event TransactionRiskUpdated(bytes32 indexed txHash, uint256 value, uint8 riskScore, bool isFlagged);
    event WalletScoreUpdated(address indexed wallet, uint8 trustScore, bool isHighRisk);
    event OracleAuthorized(address indexed oracle);
    event OracleRevoked(address indexed oracle);

    // --- Modifiers ---
    modifier onlyOwner() {
        require(msg.sender == owner, "Only owner can call this");
        _;
    }
    
    modifier onlyOracle() {
        require(authorizedOracles[msg.sender] || msg.sender == owner, "Not authorized oracle");
        _;
    }

    constructor() {
        owner = msg.sender;
        authorizedOracles[msg.sender] = true; // Deployer is auto-authorized
    }

    // =============================================================
    // 1. CORE FUNCTIONS (Single Update)
    // =============================================================

    function updateTransactionRisk(
        bytes32 txHash, 
        uint256 value, 
        uint8 riskScore
    ) external onlyOracle {
        require(riskScore <= 100, "Score > 100");

        transactions[txHash] = TransactionRecord({
            value: value,
            timestamp: uint48(block.timestamp),
            riskScore: riskScore
        });

        // Emit event so your Frontend 'Live Feed' sees it instantly
        emit TransactionRiskUpdated(txHash, value, riskScore, riskScore >= highRiskThreshold);
    }

    function updateWalletScore(address wallet, uint8 trustScore) external onlyOracle {
        require(trustScore <= 100, "Score > 100");

        wallets[wallet] = WalletInfo({
            trustScore: trustScore,
            lastUpdated: uint48(block.timestamp)
        });

        emit WalletScoreUpdated(wallet, trustScore, trustScore < minTrustScore);
    }

    // =============================================================
    // 2. REAL-LIFE FEATURE (Batch Updates)
    // =============================================================
    
    /** * @dev Updates multiple wallet scores in one transaction. 
     * Saves ~20% gas compared to calling updateWalletScore 10 times.
     */
    function batchUpdateWallets(address[] calldata walletAddrs, uint8[] calldata scores) external onlyOracle {
        require(walletAddrs.length == scores.length, "Mismatched arrays");
        
        for(uint i = 0; i < walletAddrs.length; i++) {
            wallets[walletAddrs[i]] = WalletInfo({
                trustScore: scores[i],
                lastUpdated: uint48(block.timestamp)
            });
            emit WalletScoreUpdated(walletAddrs[i], scores[i], scores[i] < minTrustScore);
        }
    }

    // =============================================================
    // 3. READ FUNCTIONS (For Frontend)
    // =============================================================

    function getWalletTrustScore(address wallet) external view returns (uint8) {
        if (wallets[wallet].lastUpdated == 0) return 50; // Default neutral score
        return wallets[wallet].trustScore;
    }

    function getTransactionRisk(bytes32 txHash) external view returns (uint8) {
        return transactions[txHash].riskScore;
    }

    function isHighRiskWallet(address wallet) external view returns (bool) {
        if (wallets[wallet].lastUpdated == 0) return false; // Unknown != High Risk
        return wallets[wallet].trustScore < minTrustScore;
    }

    // =============================================================
    // 4. ADMIN FUNCTIONS
    // =============================================================

    function setOracle(address oracle, bool isActive) external onlyOwner {
        authorizedOracles[oracle] = isActive;
        if(isActive) emit OracleAuthorized(oracle);
        else emit OracleRevoked(oracle);
    }

    function setThresholds(uint8 _minTrust, uint8 _highRisk) external onlyOwner {
        minTrustScore = _minTrust;
        highRiskThreshold = _highRisk;
    }
}