# Oracle End-to-End Validation Report (2026-04-04)

- Generated at: 2026-04-04T17:06:32.665113Z
- Scope: Live oracle fetch -> backend predict (risk+temporal+SHAP) -> on-chain write -> backend persistence -> frontend validation
- Predictions analyzed: 20

## Execution Evidence

- Backend started successfully and initialized SHAP singleton (22 features).
- Oracle processed live Sepolia batches and wrote transaction risk + wallet trust updates on-chain.
- Backend /transactions returned enriched records with temporal_state and shap_explanation payloads.

## Per-Prediction Details

### 1. 0x3c28874abb6bd2ed6c784532227e78d391f96a79eee66ceed525a002f3ae5223
- From -> To: 0x63f785f071693884c255dbe049d31fd4441cc42e -> 0xa8d080f547525a739abac0df9db8b075d1663b63
- Fraud Probability / Risk Score: 0.0889 / 1
- Trust Before -> After: 0.5 -> 0.364412
- Temporal Score: 0.364412 (x10=3.644)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.091567
- SHAP Summary: Risk is primarily increased by new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.2424,increases_risk); BMax_BMin_per_NT(-1.0546,reduces_risk); Number_of_Inputs(-0.9157,reduces_risk)
- On-chain Writes: tx_update=4244c3f6afc896769a516532299c325fb87005c6f02949caa14bb216d14f0210, wallet_update=0be0ce5ddef5e6a421c4da6ffdefb1e08c8a5bf1a2cfba248ef1e53db3b1784e

### 2. 0x0c7390e43cc3a011cf82e685a3da1b9d9a713d35b943d914155939c98785e3d7
- From -> To: 0xbe16a24f095263f98426f02cefc7cb4c0e8a05ff -> 0x2c41ed6294a15f5fbc731396fadb4723ee397f25
- Fraud Probability / Risk Score: 0.7917 / 8
- Trust Before -> After: 0.5 -> 0.111053
- Temporal Score: 0.111053 (x10=1.111)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.815451
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern and new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5848,increases_risk); Wallet_Age_Days(+1.2823,increases_risk); BMax_BMin_per_NT(-0.9057,reduces_risk)
- On-chain Writes: tx_update=db4156280cf5be9c30be63e6a5c7ced7f5f25a20187c27d9019ca47c9d71a370, wallet_update=2f9c92af0952c0060c153e83351cd05c2dd08b04b78fe8fb6aa0dab974646736

### 3. 0xc84a0a120b16eeea9cf62f1313c7de61dc4c2b1d063c9486cd1d45c293812b05
- From -> To: 0x5a27be7c11093791c8413fc9d5bfd3cbc0e11a8f -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.7513 / 8
- Trust Before -> After: 0.5 -> 0.125617
- Temporal Score: 0.125617 (x10=1.256)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.773839
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern and new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5868,increases_risk); Wallet_Age_Days(+1.2760,increases_risk); BMax_BMin_per_NT(-0.9210,reduces_risk)
- On-chain Writes: tx_update=638ae61f3e29baefdb7cd86b6ea7128f09daa98afb498ef7e8233f0eed97cc38, wallet_update=94534407765c318fa2b6172485153647d3e5f01fdf6ebac0043596fbdf4c310b

### 4. 0x96b968ed801e8568d6c421dd1a21d18fa5bd3dec3eb18a33dfed70108ed3cf97
- From -> To: 0x4b80c8b38d42d822202caff1ce8c53fd02e9340c -> 0x2c41ed6294a15f5fbc731396fadb4723ee397f25
- Fraud Probability / Risk Score: 0.796 / 8
- Trust Before -> After: 0.5 -> 0.109503
- Temporal Score: 0.109503 (x10=1.095)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.81988
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern and new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5861,increases_risk); Wallet_Age_Days(+1.2841,increases_risk); BMax_BMin_per_NT(-0.9032,reduces_risk)
- On-chain Writes: tx_update=94afbfbf2cbc7a74298c34679faf3676d25eec5e37c4f9918607c188ed38490b, wallet_update=fd5d5b65f150a8539809dbd6de579173936a017e957f9cbb48483227e2870ceb

### 5. 0x7157c7a8237a3191dd59539619fd6f36706d3c6548325f2b7f0c5e2c8809c23b
- From -> To: 0x6de30a05db4a42fb62ae2677a71e66c611218536 -> 0xa1f78bed1a79b9aec972e373e0e7f63d8cace4a8
- Fraud Probability / Risk Score: 0.0403 / 1
- Trust Before -> After: 0.5 -> 0.381933
- Temporal Score: 0.381933 (x10=3.819)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.041509
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.2253,increases_risk); Number_of_Outputs(-1.1456,reduces_risk); BMax_BMin_per_NT(-1.0984,reduces_risk)
- On-chain Writes: tx_update=65d223812dbc95cd055fa45cbed2feb096691e56af1957e26f7fa12404c5723c, wallet_update=7e0aefc7280606608b8367dbd59158d98e07b3f67c24cd14168a75e43e7d8cee

### 6. 0xdd7a27bd83ae74d0909ea7125480d83367d22efa9c9736d66597b20228ab420b
- From -> To: 0x1c1876b772485217787dd3d608a51765d8ec2981 -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.1178 / 2
- Trust Before -> After: 0.5 -> 0.353994
- Temporal Score: 0.353994 (x10=3.54)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.121334
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4753,increases_risk); Wallet_Age_Days(-2.4660,reduces_risk); BMax_BMin_per_NT(-0.7546,reduces_risk)
- On-chain Writes: tx_update=000564284c4ad524e657f9b8dd83155b224e6b6619735c9f5a933ac7b106470f, wallet_update=c341065fb8d6abf7ffd91ef78f377a780fdb8aacc4278d443057da4ed9245ed7

### 7. 0x1f0e263cd16edb35d9003f37bf450c209d5a87ad052ecf8ac5e1e7f900966b9e
- From -> To: 0x398f62f487a9138398b5fde08e07beba8698e804 -> 0x4675c7e5baafbffbca748158becba61ef3b0a263
- Fraud Probability / Risk Score: 0.0614 / 1
- Trust Before -> After: 0.5 -> 0.374326
- Temporal Score: 0.374326 (x10=3.743)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.063242
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(-1.0941,reduces_risk); Wallet_Age_Days(+1.0353,increases_risk); BMax_BMin_per_NT(-1.0129,reduces_risk)
- On-chain Writes: tx_update=ec54f0f7ac8c0eed6702e89959896be60d891f79dd7453f8773ea7b5a25c58d7, wallet_update=954151569126efc320925afeca5530152c38a82de99125f35b30b51a5f7b25fa

### 8. 0xc458dbe188cb92304a56bd7fedc30b5a2783404b2c4531288d3ef6307bfaaf42
- From -> To: 0xe95e896069beae0b75befb20a915ff57d7b488d8 -> 0x0066fe607bcf344d6e87d44795a6331fbb844717
- Fraud Probability / Risk Score: 0.046 / 1
- Trust Before -> After: 0.5 -> 0.379878
- Temporal Score: 0.379878 (x10=3.799)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.04738
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.2430,increases_risk); Number_of_Outputs(-1.1150,reduces_risk); BMax_BMin_per_NT(-1.0296,reduces_risk)
- On-chain Writes: tx_update=3f023282cae09e20959b7c5121ab3c0e0b6fe38b14c6fe47eecb9174a0e4c642, wallet_update=e677aecb3a7a5604806600a812a739cc5bb1ed488315392e01fb3220445e0126

### 9. 0xbf498480600c93ec5642f130448ac654d45b2ebf56d66b8116f784878886ed23
- From -> To: 0xea7e0753d80ddd2cc3d80716c61215eb30ae2bd8 -> 
- Fraud Probability / Risk Score: 0.0607 / 1
- Trust Before -> After: 0.5 -> 0.374578
- Temporal Score: 0.374578 (x10=3.746)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.062521
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(-1.0953,reduces_risk); Wallet_Age_Days(+1.0353,increases_risk); BMax_BMin_per_NT(-1.0129,reduces_risk)
- On-chain Writes: tx_update=5c95260f175d4cb10f63706f9a06548929f74a4e3a9dcb3004c452f2082e5dda, wallet_update=7da15f1df681e22b61ae07b0484f79494931bbdbfa2e03e736ed0e1d3cf50220

### 10. 0x1d331d4c2fbde0dfc0e2dc15905cc2c58392624fbbb1ae2e671535fac0c63e7c
- From -> To: 0xc717822490935812330bba22ceb0930b8635971c -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.1176 / 2
- Trust Before -> After: 0.5 -> 0.354066
- Temporal Score: 0.354066 (x10=3.541)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.121128
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(-2.4803,reduces_risk); Number_of_Outputs(+2.4427,increases_risk); BMax_BMin_per_NT(-0.7561,reduces_risk)
- On-chain Writes: tx_update=deb4b8898c3827c6dc30bc2332a5a7abe23261a05ffd623a759937a7090b3bb2, wallet_update=31f6cec42d2f36614ee5243e8535a7a27101bd78fe16de7903f2e1f4cd65c4f5

### 11. 0x0382a5888f02b1102f499c51947a93f684fc3e4c461702ca45ef1ea93a16e290
- From -> To: 0x7fa36d0270e1dffc4135a7cd4eaf55adcfc8d3c1 -> 0xaf93c5b321757ca9f37992525c4889bceef76726
- Fraud Probability / Risk Score: 0.1193 / 2
- Trust Before -> After: 0.5 -> 0.353453
- Temporal Score: 0.353453 (x10=3.535)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.122879
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4766,increases_risk); Wallet_Age_Days(-2.4707,reduces_risk); BMax_BMin_per_NT(-0.7532,reduces_risk)
- On-chain Writes: tx_update=afee9adb83c1b52e3c702a26b6071132796fdd2f87128fb56d88d00fa9785f1a, wallet_update=1be3a69086d31e42010ab846a3eb3e0552664c20c109cf79b8ea0e9c2bd030d2

### 12. 0xdc07281cad5c43e9b2e738eba59d8dcefa680f3945cc391cf352ea3a70782646
- From -> To: 0x0c018274fbb2ca0ffebfdcb4f13758111825c1a5 -> 0xa5c2f8f2168eb540fe9409f563ac562b931bf2ae
- Fraud Probability / Risk Score: 0.1193 / 2
- Trust Before -> After: 0.5 -> 0.353453
- Temporal Score: 0.353453 (x10=3.535)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.122879
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4766,increases_risk); Wallet_Age_Days(-2.4707,reduces_risk); BMax_BMin_per_NT(-0.7532,reduces_risk)
- On-chain Writes: tx_update=d626571962228c536f41e17116ee7be087d363db0903cc30609adf836bac6d78, wallet_update=8dd8816ecc9f85517eed61e06bac439c97dc283d60aaf0c34a295616ecdf788a

### 13. 0x8b57b476c3d12b50d74d19d740be52bc91da78a307247ea99e1e5b0646feb657
- From -> To: 0xa6026b9cf3d3088e4bf03230d9c3ea1e8bebc715 -> 0xa1f78bed1a79b9aec972e373e0e7f63d8cace4a8
- Fraud Probability / Risk Score: 0.0965 / 1
- Trust Before -> After: 0.5 -> 0.361673
- Temporal Score: 0.361673 (x10=3.617)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.099395
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5460,increases_risk); Wallet_Age_Days(-2.4725,reduces_risk); BMax_BMin_per_NT(-0.7685,reduces_risk)
- On-chain Writes: tx_update=f014f93d860c2a7ff5bf6d2cd42ddfdf9444cde842e37b2365152f6982a2e9a0, wallet_update=777b3106a6872e21083853c7467fa05772c74a7a170e9e02fdb4837ca55da642

### 14. 0x233b9fb0f1ae623e4fa6dcf6282a9c3f5320b8f07846d7d5a349b24a397a60d4
- From -> To: 0x09ef96f6751a840404cc30bce46496a2d8dbf81b -> 0x0049f503c6f4daf356cd28ef04002d5ef4a644ae
- Fraud Probability / Risk Score: 0.0503 / 1
- Trust Before -> After: 0.62 -> 0.331867
- Temporal Score: 0.331867 (x10=3.319)
- Temporal State: decay=0.0, gap=9442503s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.051809
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.1685,increases_risk); Number_of_Outputs(-1.1294,reduces_risk); BMax_BMin_per_NT(-1.0309,reduces_risk)
- On-chain Writes: tx_update=e5274f0e225fd73d7615eeaaa1b804fde395ac1b82cdbb440d8aba19d966d3b1, wallet_update=f0768645030a7a1b9de4edc738b5ad6af1591a7bd9b66aeb9bae5a534ddb5b0c

### 15. 0x3969d20bb53a494bf13ce91b007ab649304ec419f1fb0665a1a0570e4d0ba588
- From -> To: 0x7567ad5eedfff063ec615777378c0e8840b52fde -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.1166 / 2
- Trust Before -> After: 0.5 -> 0.354426
- Temporal Score: 0.354426 (x10=3.544)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.120098
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4764,increases_risk); Wallet_Age_Days(-2.4738,reduces_risk); BMax_BMin_per_NT(-0.7569,reduces_risk)
- On-chain Writes: tx_update=025cadcc14eef46c36b38d9e5e1dc9dddef5090cca503ec1d64a7cae4c59f01b, wallet_update=d1a6dc0366e9d631d6b742c81d2993dd247ae28c56297845f1ad61906ecb8ad4

### 16. 0xa768c64e3a9531f1323edd11f31d56a476f0cbea61dcbe4e74cc5b5172450f21
- From -> To: 0xe351de30dc0ec7ef6cb30dd62694de821e3ddd84 -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.1166 / 2
- Trust Before -> After: 0.5 -> 0.354426
- Temporal Score: 0.354426 (x10=3.544)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.120098
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4764,increases_risk); Wallet_Age_Days(-2.4738,reduces_risk); BMax_BMin_per_NT(-0.7569,reduces_risk)
- On-chain Writes: tx_update=125e8b0dec44fb355631e9a3dc36377de73c3fcacb8f0f9fff40780aeb29d67a, wallet_update=f69c5629c76f0efd59e2d35ea97552120b43f6f617bdfe4a93c496542eec587c

### 17. 0x10bdb00864d9df5431a2eab93885918010f643cfba81e45e38df60dc22c5d6f7
- From -> To: 0xc0db78d65481df5bc40a8eb5e8faeaf50645dcbe -> 0xcf10a5fb2ff625dfed3e513221650fe6b04d51be
- Fraud Probability / Risk Score: 0.0048 / 1
- Trust Before -> After: 0.56 -> 0.34827
- Temporal Score: 0.34827 (x10=3.483)
- Temporal State: decay=0.0, gap=10018188s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.004944
- SHAP Summary: Risk remains contained because new wallet and balance volatility pattern reduce the model output.
- SHAP Top Drivers: Wallet_Age_Days(-2.3419,reduces_risk); BMax_BMin_per_NT(-0.8760,reduces_risk); Number_of_Inputs(-0.7342,reduces_risk)
- On-chain Writes: tx_update=02a0aeb87f331e8df923b598ff107e3fb4656bf75f673de300bf312272244bbd, wallet_update=0f876b332af60de17ec6a2edb9b0861cbee4371516299168640b1f8bca73258e

### 18. 0x2c64686a5026db7d6f4c875dc9ac82d54e43b20863cc150f20251e0f706d0dcd
- From -> To: 0x95d41b376650588a8a638380b3821dc69bdde771 -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.1196 / 2
- Trust Before -> After: 0.5 -> 0.353345
- Temporal Score: 0.353345 (x10=3.533)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.123188
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4780,increases_risk); Wallet_Age_Days(-2.4646,reduces_risk); BMax_BMin_per_NT(-0.7809,reduces_risk)
- On-chain Writes: tx_update=86c56e603983b37843aaae427dc49509d4db0860f6f2ba074d56ac984798fce2, wallet_update=7e401c6531acd51caca85fb8406ca954cfba10772cbbaa1304d13ccf59f601d0

### 19. 0xcb25074d8d8a7e96981662baf7124de2974940548cc33bf31548b3088986d1c6
- From -> To: 0x79ca0e7b077cfaca7c9ddd8523bd3aeedeacf14b -> 0x79ca0e7b077cfaca7c9ddd8523bd3aeedeacf14b
- Fraud Probability / Risk Score: 0.0606 / 1
- Trust Before -> After: 0.5 -> 0.374614
- Temporal Score: 0.374614 (x10=3.746)
- Temporal State: decay=0.092922, gap=19800s, burst=False, tx_count_last_1_min=1, dormant=False, adjusted_risk=0.062418
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(-1.0941,reduces_risk); Wallet_Age_Days(+1.0399,increases_risk); BMax_BMin_per_NT(-1.0129,reduces_risk)
- On-chain Writes: tx_update=6762a3fd30f2a19a669235c3cff98959dfd63b7860495f59b07e20b79fac614b, wallet_update=e910e67b055bf2f0a7f8326c833fc9538669a4dcacab5c271e93dd9f28e5cc8c

### 20. 0x809492927c474909188bfb8729d0749b2698e70f8b43a34125a0574969ac1157
- From -> To: 0x192efeec08732bcfb18d496084a755697a4a2141 -> 0xca11bde05977b3631167028862be2a173976ca11
- Fraud Probability / Risk Score: 0.0638 / 1
- Trust Before -> After: 0.5 -> 0.56
- Temporal Score: 0.56 (x10=5.6)
- Temporal State: decay=None, gap=Nones, burst=None, tx_count_last_1_min=None, dormant=None, adjusted_risk=None
- SHAP Summary: None
- SHAP Top Drivers: 
- On-chain Writes: tx_update=055b48beb1972c28ee60f7edc64e2a6948d39a224c58ac3a6acdb300c7fe5f92, wallet_update=deaf8cadfd528eab52157a92c8c5d961331955e479d8b491ee33b842ea096122

## Frontend Validation

- Frontend renders risk/trust numeric values from riskScore and walletScore.
- Transaction modal explanation currently renders data.explanation from /predict/transaction (rule narrative), not data.shap_explanation.
- No dedicated frontend component currently renders temporal_state narrative for trust score movement.

## Conclusion

- End-to-end oracle pipeline is operational and persists SHAP + temporal enriched predictions.
- Temporal intelligence is being applied (non-default temporal scores, decay and adjusted risk observed).
- SHAP explanations are generated and stored per prediction; frontend currently surfaces scores and rule text but not full SHAP/trust-temporal explanation blocks.