# Bonn dataset inspection

`python inspect_bonn.py` (pandas, NumPy) reproduces the counts below from the project root. They describe what the published tables contain; they are not uptime or transmission logs.

## What is present

- `survey/samples.csv`: 131,057 reception rows from 2020-07-13 to 2021-03-05, on 171 distinct days, with latitude, longitude, date, packet number, RSSI, SNR and gateway ID. No field is missing and no row is an exact duplicate.
- `survey/gateways.csv`: 9 sites with coordinates and antenna heights; every gateway ID in the receptions matches one. Surface distances run from 0.218 km to 12.959 km, median 2.476 km.
- `survey/signalserver.csv`: 131,057 rows. The survey README says these rows line up with the sample rows, so keep the original order when joining. All six columns have values, but `itm` and `egli` are identical in every row, so they can't serve as two independent baselines.

## Packet identity and negative labels

The authors' notebook groups readings by date, packet number, latitude and longitude. That gives 56,540 groups and 131,057 group-gateway receptions, with no gateway repeated inside a group, which makes the grouping a reasonable working key. It isn't proof that each group is one transmission: the table has no device ID or time of day, so two transmissions with the same date, counter and position would merge.

Date and packet number alone are not enough. 3,916 of 51,357 date-counter combinations occur at more than one position; counter 0, for instance, appears at 100 positions on 2020-07-23.

Treating every gateway as active for the whole study gives 508,860 pairs and 377,803 apparent misses, which is wrong because the gateways' active periods differ (table below). Restricting each gateway to the span between its first and last reception gives 382,255 pairs and 251,198 apparent misses. Requiring at least one reception from the gateway on the same date gives 380,898 pairs and 249,841 apparent misses. These are sensitivity scenarios. A gateway can drop out within a day, first and last receptions don't show uptime, packets nobody heard are missing, and all three counts depend on the packet grouping.

| Gateway | First reception | Last reception | Days with receptions | Receptions |
|---|---|---|---:|---:|
| 015c | 2020-07-13 | 2021-03-05 | 171 | 16,422 |
| 0172 | 2020-07-15 | 2021-03-05 | 159 | 11,057 |
| 01c6 | 2020-07-13 | 2021-03-05 | 171 | 11,832 |
| 06df | 2020-07-13 | 2021-01-26 | 142 | 31,737 |
| 08d5 | 2020-10-14 | 2021-03-03 | 100 | 9,985 |
| 08fc | 2020-11-10 | 2021-03-05 | 85 | 11,110 |
| 090e | 2020-09-24 | 2021-03-05 | 117 | 10,116 |
| 0917 | 2020-09-29 | 2021-03-05 | 114 | 16,544 |
| 244d | 2020-07-13 | 2020-10-13 | 67 | 12,254 |

## Physics baseline

The authors' notebook uses 868 MHz, a 2 m sensor height and a combined link budget of `14 + 3 - 2 = 15 dB` (transmit power, receiver antenna gain, receiver cable loss). Their sensor firmware sets SF12 and 14 dBm, but the table doesn't record the settings per packet, so these are modelling assumptions.

The notebook computes received packet power as `rssi + min(snr, 0)` and compares it with the power implied by each model's path loss. The correction matters because 98,218 of the 131,057 receptions have negative SNR. Any comparison has to use one quantity consistently, and other hardware may define RSSI differently.

## How to use the labels

Start with regression on received power and leave-one-gateway-out tests, which need no miss labels. Questions only the authors could answer are gateway uptime, original timestamps, device IDs, packet-counter behaviour and the identical ITM/Egli values. Any miss labels should be called inferred and tested under more than one activity assumption. A classifier trained only on packets that another gateway heard is conditional on that, and a held-out gateway's test packets must be chosen without looking at whether it heard them. Power regression is likewise conditional on reception and says nothing directly about coverage probability.
