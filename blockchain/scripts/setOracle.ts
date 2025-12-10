// scripts/setOracle.ts
import { ethers } from "ethers";
import "dotenv/config";

// Import the contract's artifact (ABI)
import FraudDetectionArtifact from "../artifacts/contracts/TrustScore.sol/FraudDetection.json";

async function main() {
  console.log("Starting manual authorizeOracle script...");

  // 1. Get Environment Variables
  const { 
    ALCHEMY_API_KEY, 
    SEPOLIA_PRIVATE_KEY, // This is the OWNER's key (the one you deployed with)
    ORACLE_PRIVATE_KEY,   // This is the NEW ORACLE's key
    CONTRACT_ADDRESS
  } = process.env;

  if (!ALCHEMY_API_KEY || !SEPOLIA_PRIVATE_KEY || !ORACLE_PRIVATE_KEY || !CONTRACT_ADDRESS) {
    console.error("Missing one or more required .env variables (check for all 4).");
    process.exit(1);
  }
  
  const SEPOLIA_URL = `https://eth-sepolia.g.alchemy.com/v2/${ALCHEMY_API_KEY}`;

  // 2. Create Provider and Signer (The OWNER)
  const provider = new ethers.JsonRpcProvider(SEPOLIA_URL);
  const ownerSigner = new ethers.Wallet(SEPOLIA_PRIVATE_KEY, provider);

  // 3. Get the Oracle's Address (from its separate private key)
  const oracleWallet = new ethers.Wallet(ORACLE_PRIVATE_KEY);
  const oracleAddress = oracleWallet.address;
  
  console.log(`Contract Address:     ${CONTRACT_ADDRESS}`);
  console.log(`Owner/Signer Address: ${ownerSigner.address}`);
  console.log(`Authorizing New Oracle: ${oracleAddress}`);

  // 4. Get the deployed contract instance
  const contractAbi = FraudDetectionArtifact.abi;
  const fraudDetectionContract = new ethers.Contract(
    CONTRACT_ADDRESS,
    contractAbi,
    ownerSigner // We connect the contract to the OWNER's wallet
  );

  // 5. Check if oracle is already authorized
  try {
    const isAlreadyAuthorized = await fraudDetectionContract.isAuthorizedOracle(oracleAddress);
    if (isAlreadyAuthorized) {
      console.log(`✅ Oracle ${oracleAddress} is already authorized!`);
      return;
    }
  } catch (error) {
    console.log("Checking oracle authorization status...");
  }

  console.log("Calling authorizeOracle() on the contract...");
  
  // 6. Send the transaction to authorize the oracle
  const tx = await fraudDetectionContract.authorizeOracle(oracleAddress);

  console.log("Transaction sent, waiting for mining...", tx.hash);
  const receipt = await tx.wait(); // Wait for the tx to be mined
  
  console.log(`✅ Oracle address successfully authorized! Transaction mined in block: ${receipt.blockNumber}`);
  
  // 7. Verify authorization
  try {
    const isAuthorized = await fraudDetectionContract.isAuthorizedOracle(oracleAddress);
    if (isAuthorized) {
      console.log(`✅ Verification successful: Oracle ${oracleAddress} is now authorized!`);
    } else {
      console.log(`⚠️  Verification failed: Oracle ${oracleAddress} is not authorized.`);
    }
  } catch (error) {
    console.log("Could not verify authorization status.");
  }
  
  console.log("You can now run the Python backend (oracle.py).");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
