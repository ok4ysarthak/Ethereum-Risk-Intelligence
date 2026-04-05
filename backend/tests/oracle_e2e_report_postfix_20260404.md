# Oracle End-to-End Post-Fix Validation Report (2026-04-04)

- Generated (UTC): 2026-04-04T17:23:07.644904+00:00
- Source window: last 30 minutes
- Records in report: 20

## Summary Evidence

- SHAP present in report records: 20/20
- Temporal state present in report records: 20/20
- On-chain tx hash present: 20/20
- On-chain wallet hash present: 20/20
- Low-risk records (risk <= 2): 15
- Low-risk records with non-trivial trust drop (< -0.05): 2
- Non-trivial low-risk drops: [{'tx': '0x3c28874abb6bd2ed6c784532227e78d391f96a79eee66ceed525a002f3ae5223', 'delta': -0.135588}, {'tx': '0x7157c7a8237a3191dd59539619fd6f36706d3c6548325f2b7f0c5e2c8809c23b', 'delta': -0.118067}]

## Per-Prediction Details

### 1. 0x472d241194964aba71bd759b1da23fcee2e4722fb4913325a2e771fac5f8dec1
- Timestamp: 1775323386
- From -> To: 0x9996422a7c368f635e9f22b3d38448c73127cf48 -> 0xa1f78bed1a79b9aec972e373e0e7f63d8cace4a8
- Fraud Probability / Risk Score: 0.0781 / 1
- Trust Before -> After: 0.5 -> 0.538071
- Temporal Score: 0.538071 (x10=5.381)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.080443, formula_norm=0.608774, alpha=0.35
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5964,increases_risk); Wallet_Age_Days(-2.4018,reduces_risk); BMax_BMin_per_NT(-0.7770,reduces_risk); Number_of_Inputs(-0.6635,reduces_risk); Gas_Complexity(-0.6211,reduces_risk)
- On-chain Writes: tx_update=681ab8d765d13163ddbe75d18072992c03d1a84c1c0f815c6ce18aa99427a105, wallet_update=aa0f7887639bf2a97157ff5edd11b2867797a9653971e70048f887bc9ff53d1e

### 2. 0x669044b636bc2cd20db4c7578033c46e7976553dbe8cc52886ba17eba21cee63
- Timestamp: 1775323382
- From -> To: 0x8463eaf15bcade3b3484dc17a853aa1135b4482f -> 0xca11bde05977b3631167028862be2a173976ca11
- Fraud Probability / Risk Score: 0.0647 / 1
- Trust Before -> After: 0.5 -> 0.539323
- Temporal Score: 0.539323 (x10=5.393)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.066641, formula_norm=0.612352, alpha=0.35
- SHAP Summary: Risk is primarily increased by new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.2353,increases_risk); BMax_BMin_per_NT(-1.0713,reduces_risk); Number_of_Inputs(-0.9206,reduces_risk); Gas_Complexity(-0.8762,reduces_risk); Number_of_Outputs(-0.7815,reduces_risk)
- On-chain Writes: tx_update=2a2fc438cfaed3b48418dd4b9bc3f8bd1e177744ad187e6fcceefc35cba2879b, wallet_update=2f46f2a39a04700f2cdea87a03ab2fe2e7ea764a932f1b4f074791a89d039096

### 3. 0x17019d362251215aa0779ce7cdc8e9a209528a5262f9271be0e1aa2a1de66b78
- Timestamp: 1775323370
- From -> To: 0x35af12026bfdde24f12ebf3ad926fc50c3f96abf -> 0xf24b03a130a99d8aecfa1c5cbe1a313c884d72b3
- Fraud Probability / Risk Score: 0.0023 / 1
- Trust Before -> After: 0.5 -> 0.545155
- Temporal Score: 0.545155 (x10=5.452)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.002369, formula_norm=0.629015, alpha=0.35
- SHAP Summary: Risk remains contained because new wallet and mass-distribution output pattern reduce the model output.
- SHAP Top Drivers: Wallet_Age_Days(-2.1420,reduces_risk); Number_of_Outputs(-0.9376,reduces_risk); BMax_BMin_per_NT(-0.8519,reduces_risk); Number_of_Inputs(-0.7289,reduces_risk); Gas_Complexity(-0.7011,reduces_risk)
- On-chain Writes: tx_update=fc03ecaa0a414660451a53cc2b2456560e4238c8a9b728d4003487264294311b, wallet_update=7aa4c3e7b7a57d239c1ce6a41afa255609a94afe34bb2fdaf9bcf4bf2a09ab0d

### 4. 0x4c205f84abc2cc38df6607e2368fb1d2eb5e52ccecde90e7da9ef757c57bb34e
- Timestamp: 1775323353
- From -> To: 0x7e4820f9e1860c01ec9f005f1144d6ef22659e31 -> 0x00cd23de42a8bf4774b5f87ab6583b634aff1589
- Fraud Probability / Risk Score: 0.0724 / 1
- Trust Before -> After: 0.5 -> 0.538604
- Temporal Score: 0.538604 (x10=5.386)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.074572, formula_norm=0.610296, alpha=0.35
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.2445,increases_risk); Number_of_Outputs(-1.1141,reduces_risk); BMax_BMin_per_NT(-1.0101,reduces_risk); Number_of_Inputs(-0.9053,reduces_risk); Gas_Complexity(-0.8416,reduces_risk)
- On-chain Writes: tx_update=74e07a633da15082b5922a25b330147efb40be90e065cfe69748d7dbb2f5be19, wallet_update=4015b3627cc5a2ead3770e008c0d7eb56cb5a719bbe3a16c7abc6c7c7e34c5b8

### 5. 0x4e7b066aa4c573f7ffa01f032ffab69dfec032f3af43e89e77211b52d2c1677e
- Timestamp: 1775323295
- From -> To: 0xdf7c9cc44340aaf0d72bc4e7a19e0f4cf83223d9 -> 0x0064c21b0afa7e2df6171379617027794def6b18
- Fraud Probability / Risk Score: 0.0676 / 1
- Trust Before -> After: 0.56 -> 0.670485
- Temporal Score: 0.670485 (x10=6.705)
- Temporal State: decay=0.27369, gap=10798s, burst=False, dormant=False, adj_risk=0.069628, formula_norm=0.767844, alpha=0.531578
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.1929,increases_risk); Number_of_Outputs(-1.1159,reduces_risk); BMax_BMin_per_NT(-1.0128,reduces_risk); Number_of_Inputs(-0.9046,reduces_risk); Gas_Complexity(-0.8407,reduces_risk)
- On-chain Writes: tx_update=f082a92b2a2df9e8c9817952f089047c674ff30a36f518449503212b5c5c49bb, wallet_update=3cc5d9ea7360b85bf3ec7108f902b8c7a952d4fd8b59ca1b3add42ae6c4f2aa9

### 6. 0x022fb78ad64aae1bfe79c70e5884df4577f5d689ed7aaba189252fef95827289
- Timestamp: 1775323267
- From -> To: 0x1d09fe98819b6f5a46552eff51828e576b0f3b4d -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.1162 / 2
- Trust Before -> After: 0.5 -> 0.53451
- Temporal Score: 0.53451 (x10=5.345)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.119686, formula_norm=0.5986, alpha=0.35
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4764,increases_risk); Wallet_Age_Days(-2.4642,reduces_risk); BMax_BMin_per_NT(-0.7569,reduces_risk); Number_of_Inputs(-0.6610,reduces_risk); Gas_Complexity(-0.6098,reduces_risk)
- On-chain Writes: tx_update=575728c4d3f248ef6b068e94082432ade13e1a88f4714b8ce041a788f50f98cb, wallet_update=5daa31d9b31b8b7afb963ca9bb3a965a75a07c383a010d9d8ecdb54fa703d00f

### 7. 0x25f237cf35846e7eef3435d22173895940d38cadd220600ad69e4435905fbf17
- Timestamp: 1775323264
- From -> To: 0x2a884e5946cbc367950dc684aff85cdf612d3fee -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.1186 / 2
- Trust Before -> After: 0.5 -> 0.534286
- Temporal Score: 0.534286 (x10=5.343)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.122158, formula_norm=0.597959, alpha=0.35
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4780,increases_risk); Wallet_Age_Days(-2.4576,reduces_risk); BMax_BMin_per_NT(-0.7803,reduces_risk); Number_of_Inputs(-0.6647,reduces_risk); Gas_Complexity(-0.6100,reduces_risk)
- On-chain Writes: tx_update=1e883844ff07443cacb2a6ed9bcb6a4224cf03ca98a305630efdb2c7a688b1c5, wallet_update=2af50747d2a33af90356ad069a632ac64fce77c5ca1d6ab24fa7f9cec3dc5e86

### 8. 0xa6d36198263fa7dff32c72aba9bc923afcb151ded05933ddaaec3f04cb1cd5b1
- Timestamp: 1775323261
- From -> To: 0x6cc9397c3b38739dacbfaa68ead5f5d77ba5f455 -> 0xe7ec6ec7aa8a7c62e1de6a2f02e5c109ac42bf55
- Fraud Probability / Risk Score: 0.0023 / 1
- Trust Before -> After: 1.0 -> 0.97
- Temporal Score: 0.97 (x10=9.7)
- Temporal State: decay=0.0, gap=9424338s, burst=False, dormant=True, adj_risk=0.105369, formula_norm=0.894631, alpha=0.6
- SHAP Summary: Risk remains contained because new wallet and mass-distribution output pattern reduce the model output.
- SHAP Top Drivers: Wallet_Age_Days(-2.1537,reduces_risk); Number_of_Outputs(-0.9440,reduces_risk); BMax_BMin_per_NT(-0.8227,reduces_risk); Number_of_Inputs(-0.7351,reduces_risk); Gas_Complexity(-0.7067,reduces_risk)
- On-chain Writes: tx_update=956925ff4299ecf1d6a4d703628684454da8c07aaf3b0ba8541877a08ae267f8, wallet_update=f660009c6b9c7550aaac8e51dd8d3e1a24742dc24ce8bb5c931f93d3252d23bd

### 9. 0x37b715d12ef88390c42dd7e1bcedc3271548d5880d71bd5ae22b2c8d38b894d6
- Timestamp: 1775323260
- From -> To: 0x94072243e3344ae3d80509f6db2e0cb212adee79 -> 0x7c6a623ec16d436cc4efcf959ed2ef2f71a64654
- Fraud Probability / Risk Score: 0.005 / 1
- Trust Before -> After: 0.5 -> 0.544903
- Temporal Score: 0.544903 (x10=5.449)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.00515, formula_norm=0.628294, alpha=0.35
- SHAP Summary: Risk remains contained because new wallet and balance volatility pattern reduce the model output.
- SHAP Top Drivers: Wallet_Age_Days(-2.3560,reduces_risk); BMax_BMin_per_NT(-0.8731,reduces_risk); Number_of_Inputs(-0.7375,reduces_risk); Number_of_Outputs(-0.7196,reduces_risk); Gas_Complexity(-0.6928,reduces_risk)
- On-chain Writes: tx_update=c383bad018ecddff4ac662c457b290ef45d15d7370674d6ae166e3928b692cba, wallet_update=cab5bff37afc9ef828a4f8ad47c24330a02ab48fe24e2dcd3e72503c2288973b

### 10. 0x520335abfd9865c80930f5ab137972145a3a1fc3b0113791eae019faeccda1d1
- Timestamp: 1775323255
- From -> To: 0xcfdc739848ebe8a959e943df692fcf6ec91bb56c -> 0x10c15c29fb208047df6247514c0416c217e6f6e2
- Fraud Probability / Risk Score: 0.789 / 8
- Trust Before -> After: 0.5 -> 0.471628
- Temporal Score: 0.471628 (x10=4.716)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.81267, formula_norm=0.418937, alpha=0.35
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern and new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5859,increases_risk); Wallet_Age_Days(+1.2828,increases_risk); BMax_BMin_per_NT(-0.9079,reduces_risk); Number_of_Inputs(-0.8324,reduces_risk); Gas_Complexity(-0.7440,reduces_risk)
- On-chain Writes: tx_update=52fb65a85161aedf0070908eea2f2da4bd2d587640d011b8b2b91db2678d19c1, wallet_update=670d35cc1d72494c054d7ba3ca6940e5f471779ac1a84b880396903abea54623

### 11. 0x549512ec2a090ceae9b2fcc69ff115362cdb421e36dd32444213ba6daeb0eb1b
- Timestamp: 1775323252
- From -> To: 0x6bd62fd98dc7bdb0dd357c010cd656cf37399e06 -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.1173 / 2
- Trust Before -> After: 0.5 -> 0.534407
- Temporal Score: 0.534407 (x10=5.344)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.120819, formula_norm=0.598306, alpha=0.35
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(-2.4756,reduces_risk); Number_of_Outputs(+2.4753,increases_risk); BMax_BMin_per_NT(-0.7552,reduces_risk); Number_of_Inputs(-0.6618,reduces_risk); Gas_Complexity(-0.6100,reduces_risk)
- On-chain Writes: tx_update=752efdfa7f9a9d33e4aa138b17ca68bdb4082db93df69028b49d648b2e0e1a3c, wallet_update=946f4640dee91faa19ac81d4e0838b3a4fbfba444ed89f624a8f305f21a69595

### 12. 0x81989a0c06f39cccd57b285baf47e1955b307d0a30ced1f035ab59e300740ca8
- Timestamp: 1775323248
- From -> To: 0xdac509c5692f73dcc45d70558d1f4d2999d27b8c -> 0xa1f78bed1a79b9aec972e373e0e7f63d8cace4a8
- Fraud Probability / Risk Score: 0.1207 / 2
- Trust Before -> After: 0.5 -> 0.534089
- Temporal Score: 0.534089 (x10=5.341)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.124321, formula_norm=0.597398, alpha=0.35
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(-2.4805,reduces_risk); Number_of_Outputs(+2.4766,increases_risk); BMax_BMin_per_NT(-0.7528,reduces_risk); Number_of_Inputs(-0.6639,reduces_risk); Gas_Complexity(-0.6102,reduces_risk)
- On-chain Writes: tx_update=bea3d6bac437c9a8e71ee463f30caaafa4c052b8bdd09cd6821b884c5a277617, wallet_update=e6d69954ad1b4b5ef8cf53d6ea080842631ec55ea091b3de5289d7f582474323

### 13. 0x992e2942ba6bc92804439f97a7aff17c9147371e993a988c71c3cf16f9bd2181
- Timestamp: 1775323245
- From -> To: 0xa11d929e93a159608f05b73f58cf94e3d1d0ad1e -> 0x4fe3ff7d0a60da5ca44923cd0a8596df54997a82
- Fraud Probability / Risk Score: 0.0817 / 1
- Trust Before -> After: 0.5 -> 0.537734
- Temporal Score: 0.537734 (x10=5.377)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.084151, formula_norm=0.607813, alpha=0.35
- SHAP Summary: Risk is primarily increased by new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.2360,increases_risk); BMax_BMin_per_NT(-1.0554,reduces_risk); Number_of_Inputs(-0.9272,reduces_risk); Gas_Complexity(-0.8610,reduces_risk); Volatility_Age_Interaction(-0.7571,reduces_risk)
- On-chain Writes: tx_update=9bc7d04fc0352ea5b7ed3044b51d3e3e2a6cf3523941dac98d135407aef4d871, wallet_update=18481a7a87aab6b186dc07cf427e5a689c7df62270538c9d4f4ef618bbd367cd

### 14. 0xb923a38382bd07329512890c9be1e03d847184441db4ef7563f277fe0149acde
- Timestamp: 1775323242
- From -> To: 0x9919668888b4ffd3f1b78e2686d2c87454894325 -> 0xa1f78bed1a79b9aec972e373e0e7f63d8cace4a8
- Fraud Probability / Risk Score: 0.1193 / 2
- Trust Before -> After: 0.5 -> 0.53422
- Temporal Score: 0.53422 (x10=5.342)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.122879, formula_norm=0.597772, alpha=0.35
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern, while new wallet partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.4766,increases_risk); Wallet_Age_Days(-2.4709,reduces_risk); BMax_BMin_per_NT(-0.7528,reduces_risk); Number_of_Inputs(-0.6639,reduces_risk); Gas_Complexity(-0.6095,reduces_risk)
- On-chain Writes: tx_update=fcc94d06e1b4572e23161c590630e7eefc5ca1ee1c3e17dc6f8c4e70da2e1d20, wallet_update=3b9cd0a60d353ea1da2b072468d0713e1f1ee24eef0d9c14e4a83ba6a0338e22

### 15. 0x0fb082715ea7716596d9d6aa6ff0d903a15a4105f6cf0f2d87fabe17bc03e4c5
- Timestamp: 1775323238
- From -> To: 0xfb3eb3f4c936dd0afad452c34d793115a569576e -> 0xa1f78bed1a79b9aec972e373e0e7f63d8cace4a8
- Fraud Probability / Risk Score: 0.794 / 8
- Trust Before -> After: 0.5 -> 0.471161
- Temporal Score: 0.471161 (x10=4.712)
- Temporal State: decay=1.0, gap=0s, burst=False, dormant=False, adj_risk=0.81782, formula_norm=0.417602, alpha=0.35
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern and new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5830,increases_risk); Wallet_Age_Days(+1.2752,increases_risk); BMax_BMin_per_NT(-0.9059,reduces_risk); Number_of_Inputs(-0.8323,reduces_risk); Gas_Complexity(-0.7426,reduces_risk)
- On-chain Writes: tx_update=9d58296a84a7635992fa230db2048b5506f21dcdeaf1f65f6d2c148534402a62, wallet_update=2af54b3fb79e039338e8474e7c88367c1977e9549dcc453ee21ee3affb0cdba2

### 16. 0x3c28874abb6bd2ed6c784532227e78d391f96a79eee66ceed525a002f3ae5223
- Timestamp: 1775322320
- From -> To: 0x63f785f071693884c255dbe049d31fd4441cc42e -> 0xa8d080f547525a739abac0df9db8b075d1663b63
- Fraud Probability / Risk Score: 0.0889 / 1
- Trust Before -> After: 0.5 -> 0.364412
- Temporal Score: 0.364412 (x10=3.644)
- Temporal State: decay=0.092922, gap=19800s, burst=False, dormant=False, adj_risk=0.091567, formula_norm=None, alpha=None
- SHAP Summary: Risk is primarily increased by new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.2424,increases_risk); BMax_BMin_per_NT(-1.0546,reduces_risk); Number_of_Inputs(-0.9157,reduces_risk); Gas_Complexity(-0.8625,reduces_risk); Number_of_Outputs(-0.7658,reduces_risk)
- On-chain Writes: tx_update=4244c3f6afc896769a516532299c325fb87005c6f02949caa14bb216d14f0210, wallet_update=0be0ce5ddef5e6a421c4da6ffdefb1e08c8a5bf1a2cfba248ef1e53db3b1784e

### 17. 0x0c7390e43cc3a011cf82e685a3da1b9d9a713d35b943d914155939c98785e3d7
- Timestamp: 1775322306
- From -> To: 0xbe16a24f095263f98426f02cefc7cb4c0e8a05ff -> 0x2c41ed6294a15f5fbc731396fadb4723ee397f25
- Fraud Probability / Risk Score: 0.7917 / 8
- Trust Before -> After: 0.5 -> 0.111053
- Temporal Score: 0.111053 (x10=1.111)
- Temporal State: decay=0.092922, gap=19800s, burst=False, dormant=False, adj_risk=0.815451, formula_norm=None, alpha=None
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern and new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5848,increases_risk); Wallet_Age_Days(+1.2823,increases_risk); BMax_BMin_per_NT(-0.9057,reduces_risk); Number_of_Inputs(-0.8340,reduces_risk); Gas_Complexity(-0.7441,reduces_risk)
- On-chain Writes: tx_update=db4156280cf5be9c30be63e6a5c7ced7f5f25a20187c27d9019ca47c9d71a370, wallet_update=2f9c92af0952c0060c153e83351cd05c2dd08b04b78fe8fb6aa0dab974646736

### 18. 0xc84a0a120b16eeea9cf62f1313c7de61dc4c2b1d063c9486cd1d45c293812b05
- Timestamp: 1775322305
- From -> To: 0x5a27be7c11093791c8413fc9d5bfd3cbc0e11a8f -> 0xfa6419a3d3503a016df3a59f690734862ca2a78d
- Fraud Probability / Risk Score: 0.7513 / 8
- Trust Before -> After: 0.5 -> 0.125617
- Temporal Score: 0.125617 (x10=1.256)
- Temporal State: decay=0.092922, gap=19800s, burst=False, dormant=False, adj_risk=0.773839, formula_norm=None, alpha=None
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern and new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5868,increases_risk); Wallet_Age_Days(+1.2760,increases_risk); BMax_BMin_per_NT(-0.9210,reduces_risk); Number_of_Inputs(-0.8300,reduces_risk); Gas_Complexity(-0.7491,reduces_risk)
- On-chain Writes: tx_update=638ae61f3e29baefdb7cd86b6ea7128f09daa98afb498ef7e8233f0eed97cc38, wallet_update=94534407765c318fa2b6172485153647d3e5f01fdf6ebac0043596fbdf4c310b

### 19. 0x96b968ed801e8568d6c421dd1a21d18fa5bd3dec3eb18a33dfed70108ed3cf97
- Timestamp: 1775322303
- From -> To: 0x4b80c8b38d42d822202caff1ce8c53fd02e9340c -> 0x2c41ed6294a15f5fbc731396fadb4723ee397f25
- Fraud Probability / Risk Score: 0.796 / 8
- Trust Before -> After: 0.5 -> 0.109503
- Temporal Score: 0.109503 (x10=1.095)
- Temporal State: decay=0.092922, gap=19800s, burst=False, dormant=False, adj_risk=0.81988, formula_norm=None, alpha=None
- SHAP Summary: Risk is primarily increased by mass-distribution output pattern and new wallet, while balance volatility pattern partially offsets the risk.
- SHAP Top Drivers: Number_of_Outputs(+2.5861,increases_risk); Wallet_Age_Days(+1.2841,increases_risk); BMax_BMin_per_NT(-0.9032,reduces_risk); Number_of_Inputs(-0.8354,reduces_risk); Gas_Complexity(-0.7462,reduces_risk)
- On-chain Writes: tx_update=94afbfbf2cbc7a74298c34679faf3676d25eec5e37c4f9918607c188ed38490b, wallet_update=fd5d5b65f150a8539809dbd6de579173936a017e957f9cbb48483227e2870ceb

### 20. 0x7157c7a8237a3191dd59539619fd6f36706d3c6548325f2b7f0c5e2c8809c23b
- Timestamp: 1775322300
- From -> To: 0x6de30a05db4a42fb62ae2677a71e66c611218536 -> 0xa1f78bed1a79b9aec972e373e0e7f63d8cace4a8
- Fraud Probability / Risk Score: 0.0403 / 1
- Trust Before -> After: 0.5 -> 0.381933
- Temporal Score: 0.381933 (x10=3.819)
- Temporal State: decay=0.092922, gap=19800s, burst=False, dormant=False, adj_risk=0.041509, formula_norm=None, alpha=None
- SHAP Summary: Risk is primarily increased by new wallet, while mass-distribution output pattern partially offsets the risk.
- SHAP Top Drivers: Wallet_Age_Days(+1.2253,increases_risk); Number_of_Outputs(-1.1456,reduces_risk); BMax_BMin_per_NT(-1.0984,reduces_risk); Number_of_Inputs(-0.9249,reduces_risk); Gas_Complexity(-0.8603,reduces_risk)
- On-chain Writes: tx_update=65d223812dbc95cd055fa45cbed2feb096691e56af1957e26f7fa12404c5723c, wallet_update=7e0aefc7280606608b8367dbd59158d98e07b3f67c24cd14168a75e43e7d8cee

## Interpretation for Research Use

- Backend evidence supports a functioning end-to-end pipeline (oracle fetch, model infer, temporal update, SHAP, persistence, on-chain write hashes).
- In this post-fix sample, low-risk transactions do not show the earlier severe trust collapse pattern.
- SHAP is now visible in the UI transaction modal and is present in backend payloads for reproducible inspection.
- This remains observational testnet validation, not a controlled benchmark; causal claims should be avoided.