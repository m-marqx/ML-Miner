import { Core } from "@walletconnect/core";
import { WalletKit } from "@reown/walletkit";
import { projectId } from "../config/appkit"

const core = new Core({
    projectId: projectId,
});

const metadata = {
    name: 'Quantitative System',
    description: 'A quantitative trading system for crypto assets',
    url: 'https://archiemarques.com/',
    icons: ['https://avatars.githubusercontent.com/u/124513922']
}
// Create an async function to initialize the wallet kit
export const initWalletKit = async () => {
    return await WalletKit.init({
        core,
        metadata: metadata,
    });
};

// Create a promise that will resolve to the wallet kit
export const walletProvider = initWalletKit();