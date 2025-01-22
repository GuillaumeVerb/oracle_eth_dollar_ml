const hre = require("hardhat");

async function main() {
  console.log("Deploying HybridOracle contract...");

  // Déployer le contrat
  const HybridOracle = await hre.ethers.getContractFactory("HybridOracle");
  const hybridOracle = await HybridOracle.deploy();

  await hybridOracle.deployed();

  console.log(`HybridOracle deployed to: ${hybridOracle.address}`);
  
  // Attendre quelques blocks pour la vérification
  console.log("Waiting for 5 blocks for verification...");
  await hybridOracle.deployTransaction.wait(5);

  // Vérifier le contrat sur Etherscan
  console.log("Verifying contract on Etherscan...");
  try {
    await hre.run("verify:verify", {
      address: hybridOracle.address,
      constructorArguments: [],
    });
    console.log("Contract verified on Etherscan!");
  } catch (error) {
    console.error("Error verifying contract:", error);
  }
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error(error);
    process.exit(1);
  }); 