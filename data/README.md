# Data

The two radio datasets were cloned here unchanged on 2026-10-24 and are git-ignored. To get them:

```
git clone --depth 1 https://github.com/mclab-hbrs/lora-bonn data/lora-bonn
git clone --depth 1 https://github.com/oliveiraleo/LoRa-RSSI-dataset-outdoor data/LoRa-RSSI-dataset-outdoor
```

[bonn-inspection.md](bonn-inspection.md) has each gateway's active dates, the problems with identifying packets, and a duplicated propagation-model column. The missed-link counts below are hypothetical pairs, not verified failures.

## lora-bonn (primary)

[github.com/mclab-hbrs/lora-bonn](https://github.com/mclab-hbrs/lora-bonn). Rademacher, Linka, Horstmann, Henze, "Path Loss in Urban LoRa Networks: A Large-Scale Measurement Study", VTC2021-Fall ([arXiv 2109.07768](https://arxiv.org/abs/2109.07768)). Hochschule Bonn-Rhein-Sieg with Fraunhofer FKIE. Data licence Germany, attribution 2.0.

Bonn is a mid-sized city in western Germany (North Rhine-Westphalia). GPS sensors rode on the city's garbage trucks and sent LoRaWAN packets that the city's gateways logged.

| File | Contents |
|---|---|
| `survey/samples.csv` | 131,057 rows, one per gateway reception: `latitude, longitude, transceived_at (date), pkt_number, rssi, snr, gw` |
| `survey/gateways.csv` | 9 gateways: `id, latitude, longitude, antenna_height` (12-55 m) |
| `survey/signalserver.csv` | Row-aligned with `samples.csv`: predicted path loss (dB) from 6 classical models: `ecc33, cost_hata, ericcson, egli, itm, itwom` |
| `survey/analyze.ipynb` | The authors' analysis notebook (path loss vs. models) |
| `html/gateways.json` | Gateway names (depots, fire station, substations, a tower) |
| `html/data.json` | 43 MB, data behind the online hexbin map |
| `rpp-gateway-verification/` | Lab check of gateway RSSI/SNR reporting against a calibrated attenuator |
| `sensor/` | Tracker firmware and schematics (not used here) |

### Numbers

- Dates: 2020-07-13 to 2021-03-05 (171 distinct days).
- Area: about 19 km north-south by 13 km east-west.
- RSSI -130 to -66 dBm (median -119); SNR -21.8 to 12.5 dB (median -6.5).
- 56,540 distinct packets, keyed on `pkt_number + date + lat + lon`. `pkt_number` alone repeats because the counters reset.

| Gateways that heard the packet | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Packets | 16,158 | 19,090 | 12,149 | 6,154 | 2,337 | 597 | 51 | 4 |

| Gateway | 06df | 0917 | 015c | 244d | 01c6 | 08fc | 0172 | 090e | 08d5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Receptions | 31,737 | 16,544 | 16,422 | 12,254 | 11,832 | 11,110 | 11,057 | 10,116 | 9,985 |

### What this means for the project

- Gateway locations and antenna heights are included.
- Missed links can only be inferred. All 9 × 56,540 = 508,860 gateway-packet pairs split into 131,057 receptions and 377,803 apparent misses, but that assumes all nine gateways ran for the whole study, and their observed date ranges say otherwise. Packets heard by no gateway never appear, so the worst spots are under-represented. A miss should only count if the gateway was running that day; names like "HKW Nord (Old)" suggest some sites changed.
- `signalserver.csv` gives baseline predictions from ITM (Longley-Rice), COST-Hata, ECC-33, Ericsson, Egli and ITWOM, although the ITM and Egli columns are identical in every row.
- There is no spreading-factor or transmit-power column, so the paper's constant settings are assumed.

## LoRa-RSSI-dataset-outdoor (secondary)

[github.com/oliveiraleo/LoRa-RSSI-dataset-outdoor](https://github.com/oliveiraleo/LoRa-RSSI-dataset-outdoor). Oliveira, Chaves, Silva, SBSeg 2023. CC BY-NC-SA 4.0 (non-commercial).

Federal University of Juiz de Fora campus, Brazil, November-December 2022. One gateway at -21.77884, -43.37336, 932 m above sea level; no antenna height is given.

| Survey folder | Rows | Notes |
|---|---|---|
| `campus_by_car` | 498 | moving |
| `campus_walking_by_foot` | 511 | moving |
| `static_LOS_near_GW` | 496 | fixed point, line of sight |
| `static_LOS_away_from_GW` | 501 | fixed point, line of sight |
| `static_NLOS` | 508 | fixed point, blocked |

Each `*_Full-Collected-data.csv` has `Time, GPS Time, id, Latitude, Longitude, Altitude, GPS Precision, # Satellites, ED RSSI, GW RSSI`. The campus spans a few hundred metres with one gateway, so it could test whether the coverage model transfers but not placement.
