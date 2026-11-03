"""Inspect Bonn reception records without modifying the source data."""
from pathlib import Path
import numpy as np
import pandas as pd

root = Path(__file__).parent / 'data' / 'lora-bonn' / 'survey'
s = pd.read_csv(root / 'samples.csv', dtype={'gw': str})
g = pd.read_csv(root / 'gateways.csv', dtype={'id': str})
p = pd.read_csv(root / 'signalserver.csv')
key = ['transceived_at', 'pkt_number', 'latitude', 'longitude']
print('ROWS', len(s), 'PREDICTIONS', len(p), 'GATEWAYS', len(g))
print('MISSING', s.isna().sum().to_dict(), 'PRED', p.isna().sum().to_dict())
print('FULL DUPLICATES', int(s.duplicated().sum()))
print('UNMATCHED GATEWAYS', sorted(set(s.gw)-set(g.id)))
print('DATE', s.transceived_at.min(), s.transceived_at.max(), 'DAYS', s.transceived_at.nunique())
print('RSSI', s.rssi.describe().round(2).to_dict())
print('SNR', s.snr.describe().round(2).to_dict())
print('GATEWAY ACTIVITY')
print(s.groupby('gw').agg(rows=('gw','size'), first=('transceived_at','min'), last=('transceived_at','max'), days=('transceived_at','nunique')).to_string())
k = s.groupby(key, dropna=False).agg(rows=('gw','size'), gateways=('gw','nunique')).reset_index()
print('PROPOSED PACKET KEYS', len(k))
print('RECEIVERS PER KEY', k.gateways.value_counts().sort_index().to_dict())
print('KEYS WITH REPEATED GATEWAY', int((k.rows > k.gateways).sum()), 'EXTRA ROWS', int((k.rows-k.gateways).sum()))
locs = s[key].drop_duplicates().groupby(['transceived_at','pkt_number']).size()
print('DATE+PACKET KEYS', len(locs), 'WITH MULTIPLE COORDINATES', int((locs>1).sum()), 'MAX COORDINATES', int(locs.max()))
print('DATE+PACKET EXAMPLES')
print(locs[locs>1].sort_values(ascending=False).head(5).to_string())
dates = pd.to_datetime(s.transceived_at)
daily = pd.crosstab(s.transceived_at, s.gw)
print('DAYS WITH ZERO RECEPTIONS BY GATEWAY ON ANY STUDY DAY', (daily == 0).sum().to_dict())
print('ACTIVE GW PER STUDY DAY', (daily > 0).sum(axis=1).value_counts().sort_index().to_dict())
packets = s[key].drop_duplicates()
activity = s.groupby('gw').agg(first=('transceived_at','min'), last=('transceived_at','max')).reset_index()
in_range = packets.merge(activity, how='cross')
in_range = in_range[(in_range.transceived_at >= in_range['first']) & (in_range.transceived_at <= in_range['last'])]
same_day = packets.merge(s[['transceived_at','gw']].drop_duplicates(), on='transceived_at')
print('PAIR COUNTS naive/range/same-day', len(packets)*len(g), len(in_range), len(same_day))
print('POTENTIAL MISSES naive/range/same-day', len(packets)*len(g)-len(s), len(in_range)-len(s), len(same_day)-len(s))
print('MODEL STATS')
print(p.describe().loc[['min','50%','max']].round(2).to_string())
v=s.merge(g, left_on='gw',right_on='id',suffixes=('_s','_g'),validate='many_to_one')
lat1=np.radians(v.latitude_s); lat2=np.radians(v.latitude_g)
dlat=lat2-lat1; dlon=np.radians(v.longitude_g-v.longitude_s)
a=np.sin(dlat/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
km=6371*2*np.arcsin(np.sqrt(np.clip(a,0,1)))
print('DISTANCE KM QUANTILES', km.quantile([0,.01,.5,.99,1]).round(3).to_dict())
print('SIGNALSERVER IDENTICAL ITM/ITWOM', int((p.itm == p.itwom).sum()), 'ITM/EGLI', int((p.itm == p.egli).sum()))
