import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from os.path import splitext

kcfs_to_tafd = 2.29568411*10**-5 * 86400


def water_day(d):
  return d - 274 if d >= 274 else d + 91

def calc_obs_medians(df):

  df['dowy'] = np.array([water_day(d) for d in df.index.dayofyear])

  df_median = pd.DataFrame(index=df.index, columns=df.columns)
  df_median['dowy'] = df['dowy']

  for dowy in range(1,365):
    for k in df.columns:
      if k == 'dowy': continue
      m = np.median(df.loc[df.dowy==dowy, k])
      df_median.loc[df_median.dowy == dowy, k] = m

  # leap years
  for k in df.columns:
    if k == 'dowy': continue
    m = np.median(df.loc[df.dowy.isin([364,365]), k])
    df_median.loc[df_median.dowy == 365, k] = m

  return df_median

# forecasts must be (date, trace, lead)
# helper functions for baseline/perfect cases

def get_baseline_forecast(Q, Q_median, NL):
  T = len(Q)
  Qf = np.zeros((T, 1, NL))
  for t in range(T - NL):
    Qf[t,0,:] = Q_median[(t+1) : (t+1+NL)]
  return Qf


def get_perfect_forecast(Q, NL):
  T = len(Q)
  Qf = np.zeros((T, 1, NL))
  for t in range(T - NL):
    Qf[t,0,:] = Q[(t+1) : (t+1+NL)]
  return Qf

#### Alex add LT
def get_perfect_forecast_LT(Q, Q_median, NL, LT):
  T = len(Q)
  Qf = np.zeros((T, 1, NL))
  for t in range(T - NL):
    Qf[t,0,:] = Q[(t+1) : (t+1+NL)]
    Qf[t,0,LT::] = Q_median[(t+1+LT) : (t+1+NL)]
    
  return Qf



