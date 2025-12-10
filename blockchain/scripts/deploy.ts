// scripts/deploy.ts
import { ethers } from "ethers";
import "dotenv/config";
import hre from "hardhat";
import fs from "fs";
import path from "path";

// Import the contract's artifact
import FraudDetectionArtifact from "../artifacts/contracts/TrustScore.sol/FraudDetection.json";

async function main() {
  console.log("Starting manual deployment script...");

  // 1. Get Environment Variables
  const { ALCHEMY_API_KEY, SEPOLIA_PRIVATE_KEY } = process.env;

  if (!ALCHEMY_API_KEY || !SEPOLIA_PRIVATE_KEY) {
    console.error("Please set ALCHEMY_API_KEY and SEPOLIA_PRIVATE_KEY in your .env file");
    process.exit(1);
  }
  
  const SEPOLIA_URL = `https://eth-sepolia.g.alchemy.com/v2/${ALCHEMY_API_KEY}`;

  // 2. Create Provider and Signer
  const provider = new ethers.JsonRpcProvider(SEPOLIA_URL);
  const signer = new ethers.Wallet(SEPOLIA_PRIVATE_KEY, provider);

  console.log(`Deploying contract from account: ${signer.address}`);
  console.log(`Account balance: ${ethers.formatEther(await provider.getBalance(signer.address))} ETH`);

  // 3. Get Contract ABI and Bytecode
  const contractAbi = FraudDetectionArtifact.abi;
  const contractBytecode = FraudDetectionArtifact.bytecode;

  // 4. Create ContractFactory
  const fraudDetectionFactory = new ethers.ContractFactory(
    contractAbi,
    contractBytecode,
    signer
  );

  // 5. Deploy the Contract
  console.log("Deploying FraudDetection contract...");
  const fraudDetection = await fraudDetectionFactory.deploy();

  // 6. Wait for Deployment
  await fraudDetection.waitForDeployment();

  const contractAddress = await fraudDetection.getAddress();
  console.log("✅ FraudDetection contract deployed to:", contractAddress);
  
  // 7. Verify deployment by calling a view function
  try {
    const owner = await fraudDetection.owner();
    console.log("Contract owner:", owner);
    console.log("Deployment successful!");
  } catch (error) {
    console.error("Error verifying deployment:", error);
  }

  // 8. Save contract address to .env file
  saveContractAddress(contractAddress);
  
  console.log("----------------------------------------------------");
  console.log("Deployment completed successfully!");
  console.log("Save this address! You will need it for the oracle and API.");
}

function saveContractAddress(address: string) {
  const envPath = path.resolve(__dirname, "../.env");
  
  // Read existing .env file
  let envContent = "";
  if (fs.existsSync(envPath)) {
    envContent = fs.readFileSync(envPath, "utf8");
  }
  
  // Check if CONTRACT_ADDRESS already exists
  if (envContent.includes("CONTRACT_ADDRESS=")) {
    // Update existing CONTRACT_ADDRESS
    envContent = envContent.replace(
      /CONTRACT_ADDRESS=.*/g,
      `CONTRACT_ADDRESS="${address}"`
    );
  } else {
    // Add new CONTRACT_ADDRESS
    envContent += `\nCONTRACT_ADDRESS="${address}"\n`;
  }
  
  // Write back to .env file
  fs.writeFileSync(envPath, envContent);
  console.log("✅ Contract address saved to .env file");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
