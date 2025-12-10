import type { HardhatUserConfig } from "hardhat/config";
import hardhatToolboxViemPlugin from "@nomicfoundation/hardhat-toolbox-viem";

// --- FIX 1: Make sure dotenv/config is loaded at the top ---
import "dotenv/config"; 

// --- FIX 2: Re-add the import for configVariable ---
import { configVariable } from "hardhat/config";

const config: HardhatUserConfig = {
  plugins: [hardhatToolboxViemPlugin],
  solidity: {
    profiles: {
      default: {
        version: "0.8.28",
      },
      production: {
        version: "0.8.28",
        settings: {
          optimizer: {
            enabled: true,
            runs: 200,
          },
        },
      },
    },
  },
  networks: {
    hardhatMainnet: {
      type: "edr-simulated",
      chainType: "l1",
    },
    hardhatOp: {
      type: "edr-simulated",
      chainType: "op",
    },
    sepolia: {
      type: "http",
      chainType: "l1",
      
      // --- FIX 3: Use configVariable (which will now find the .env var) ---
      url: configVariable("SEPOLIA_RPC_URL"),
      accounts: [configVariable("SEPOLIA_PRIVATE_KEY")],
    },
  },

  etherscan: {
    // --- FIX 4: Use configVariable here too for consistency ---
    apiKey: configVariable("ETHERSCAN_API_KEY")
  }
};

export default config;



