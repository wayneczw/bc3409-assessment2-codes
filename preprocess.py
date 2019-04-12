import pandas as pd
import datetime
import numpy as np
import multiprocessing as mp

from multiprocessing import cpu_count

from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob

ncores = cpu_count()


def make_multi_thread(func, series):
    with mp.Pool(ncores) as pool:
        X = pool.imap(func, series, chunksize=1)
        X = [x for x in X]
    #end with
    return pd.Series(X)
#end def


def get_polarity(text):
    textblob = TextBlob(text)
    pol = textblob.sentiment.polarity
    return round(pol, 2)
#end def


def get_subjectivity(text):
    textblob = TextBlob(text)
    subj = textblob.sentiment.subjectivity
    return round(subj, 2)
#end def


########################## DJI Preprocessing
df = pd.read_csv('data.csv')
df['Date'] = pd.to_datetime(df['Date'], format="%d/%m/%Y")
df = df[df['Date'] >= datetime.date(1999, 2, 8)].reset_index(drop=True)

df = df.drop(['Close'], axis=1)
print(df.describe())
print(df.shape)

print(df[df['Adj Close'] == 0])
# 1398 2004-08-31   0.0   0.0  0.0        0.0  163580000
# 2274 2008-02-25   0.0   0.0  0.0        0.0  288600000
# 4272 2016-02-01   0.0   0.0  0.0        0.0  114450000

print(df[df['Adj Close'] == 1000000])
# 440  2000-11-02   100000.0   1000000.0  1000000.0  1000000.0  250440000
# 3330 2012-05-02  1000000.0   1000000.0  1000000.0  1000000.0  100770000
# 4669 2017-08-28   800002.5  10000000.0  1000000.0  1000000.0  218740000

########## Replace wrong data
tmp_df = df[~df['Adj Close'].isin([0, 1000000])]

open_avg_diff = pd.Series.mean(tmp_df['Open'] - tmp_df['Adj Close'])
high_avg_diff = pd.Series.mean(tmp_df['High'] - tmp_df['Adj Close'])
low_avg_diff = pd.Series.mean(tmp_df['Low'] - tmp_df['Adj Close'])

# https://www.tradingview.com/chart/?symbol=DJ%3ADJI
df.loc[1398, 'Adj Close'] = 10173.92  # from official web
df.loc[1398, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[1398, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[1398, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[2274, 'Adj Close'] = 12570.22  # from official web
df.loc[2274, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[2274, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[2274, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[4272, 'Adj Close'] = 16449.18  # from official web
df.loc[4272, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[4272, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[4272, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[440, 'Adj Close'] = 10880.51  # from official web
df.loc[440, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[440, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[440, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[3330, 'Adj Close'] = 13268.57  # from official web
df.loc[3330, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[3330, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[3330, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

df.loc[4669, 'Adj Close'] = 21808.4  # from official web
df.loc[4669, 'Open'] = pd.Series.mean(tmp_df['Open'] + open_avg_diff)
df.loc[4669, 'High'] = pd.Series.mean(tmp_df['High'] + high_avg_diff)
df.loc[4669, 'Low'] = pd.Series.mean(tmp_df['Low'] + low_avg_diff)

del tmp_df

print(df.describe())
cont_features = [
    'Open', 'High',
    'Low', 'Adj Close',
    'Volume']

########### Scaling
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled = scaler.fit_transform(df[cont_features])
print(scaled)

df[cont_features] = pd.DataFrame(scaled)

########### Get date info
df['Day of Week'] = df.apply(lambda x: x['Date'].weekday(), axis=1)
df['Day of Week'] = df['Day of Week'].astype('category')

df['Month of Year'] = df.apply(lambda x: x['Date'].month, axis=1)
df['Month of Year'] = df['Month of Year'].astype('category')

df['Day of Month'] = df.apply(lambda x: x['Date'].day, axis=1)
df['Day of Month'] = df['Day of Month'].astype('category')

df = df[df['Date'] >= datetime.date(2002, 1, 2)].reset_index(drop=True)
df = df[df['Date'] <= datetime.date(2015, 12, 31)].reset_index(drop=True)

########################## Crude Oil Price Preprocessing
cdf = pd.read_csv('./Energy_Price.csv')
cdf = cdf.dropna()

########### Compute Price Change
size = cdf.shape[0]
output_list = []
for i, row in cdf.iterrows():
    cur = row['Crude Oil']
    if (i == 0):
        output_list.append(np.nan)
        prev = cur
        continue
    else:
        if cur > prev:
            output_list.append(1)
        else:
            output_list.append(0)

    prev = cur
#end for
cdf['PChange'] = pd.Series(output_list)

cdf['Date'] = cdf['Date'].apply(lambda x: x[:-2] + '20' + x[-2:])
cdf['Date'] = pd.to_datetime(cdf['Date'], format="%d/%m/%Y")

new_df = df.merge(cdf, on='Date')

'''
########################## News Headline Preprocessing
news_df = pd.read_csv('./abcnews-date-text.csv')
news_df['Date'] = pd.to_datetime(news_df['Date'])
news_polarity = news_df.groupby('Date')['polarity'].mean()
news_subjectivity = news_df.groupby('Date')['subjectivity'].mean()

new_df = new_df.merge(news_polarity, on='Date')
new_df = new_df.merge(news_subjectivity, on='Date')

# the following chunk will get POLARITY and SUBJECTIVITY scores of the headlines
# and they only need to be run once
# news_df['publish_date'] = news_df['publish_date'].astype(str)
# news_df['Date'] = pd.to_datetime(news_df['publish_date'], format="%Y%m%d")
# news_df = news_df.drop('publish_date', axis=1)
# news_df['polarity'] = make_multi_thread(get_polarity, news_df['headline_text'])
# news_df['subjectivity'] = make_multi_thread(get_subjectivity, news_df['headline_text'])
# news_df.to_csv('abcnews-date-text.csv', index=False)
'''


########################## Irish Headline Preprocessing

# keep_news_cat = [
#     'business', 'news', 'opinion', 'opinion.letters',
#     'news.politics.oireachtas', 'culture.media', 'news.world.europe',
#     'news.health', 'news.education',
#     'business.commercial-property',
#     'news.consumer', 'news.science', 'news.law',
#     'news.technology.game-reviews', 'news.social-affairs.religion-and-beliefs',
#     'business.technology', 'business.economy.ireland',
#     'business.economy.public-finances', 'business.energy-and-resources',
#     'business.financial-services', 'business.retail-and-services',
#     'business.transport-and-tourism', 'business.economy.europe',
#     'business.markets.equities',
#     'business.economy.world', 'business.media-and-marketing',
#     'business.manufacturing', 'business.economy', 'business.markets',
#     'business.markets.bonds', 'business.economy.employment',
#     'business.agribusiness-and-food', 'business.markets.market-news',
#     'business.markets.commodities',
#     'news.technology', 'news.world.uk', 'news.world',
#     'news.social-affairs', 'news.world.middle-east', 'news.politics',
#     'news.world.us',
#     'news.world.asia-pacific', 'opinion.columnists',
#     'news.world.africa', 'business.exchange-rates',
#     'business.personal-finance',
#     'news.ireland',
#     'news.ireland.irish-news', 'news.law.courts',
#     'business.innovation',
#     'opinion.opinion-analysis',
#     'business.companies',
#     'business.technology.gadgets',
#     'business.aib-start-up-academy',
#     'business.work',
#     'business.technology.web-summit',
#     'news.law.courts.coroner-s-court', 'news.law.courts.district-court',
#     'news.law.courts.high-court', 'news.law.courts.circuit-court',
#     'business.construction', 'news.law.courts.criminal-court',
#     'news.law.courts.supreme-court',
#     'business.panama-papers',
#     'news.education.examtimes', 'news.science.space']

irish_df = pd.read_csv('./irishtimes-date-text.csv')
irish_df['Date'] = pd.to_datetime(irish_df['Date'])
irish_polarity = irish_df.groupby('Date')['polarity'].mean()
irish_subjectivity = irish_df.groupby('Date')['subjectivity'].mean()

new_df = new_df.merge(irish_polarity, on='Date')
new_df = new_df.merge(irish_subjectivity, on='Date')

'''
irish_df = irish_df.dropna()

# the following chunk will get POLARITY and SUBJECTIVITY scores of the headlines
# and they only need to be run once
# irish_df['publish_date'] = irish_df['publish_date'].astype(str)
# irish_df['Date'] = pd.to_datetime(irish_df['publish_date'], format="%Y%m%d")
# irish_df = irish_df.drop('publish_date', axis=1)
# irish_df['polarity'] = make_multi_thread(get_polarity, irish_df['headline_text'])
# irish_df['subjectivity'] = make_multi_thread(get_subjectivity, irish_df['headline_text'])
# irish_df.to_csv('irishtimes-date-text.csv', index=False)

irish_df = irish_df[irish_df['headline_category'].isin(keep_news_cat)]
'''

########################## Exchange Rate Preprocessing
ex_df = pd.read_csv('./exchange_rates.csv')
ex_df['Date'] = pd.to_datetime(ex_df['Date'])
new_df = new_df.merge(ex_df, on='Date')

'''
# print(ex_df.columns)
# ['RXI$US_N.B.EU', 'RXI$US_N.B.UK', 'RXI_N.B.BZ', 'RXI_N.B.CH',
#        'RXI_N.B.DN', 'RXI_N.B.IN', 'RXI_N.B.JA', 'RXI_N.B.KO', 'RXI_N.B.MA',
#        'RXI_N.B.MX', 'RXI_N.B.NO', 'RXI_N.B.SD', 'RXI_N.B.SF', 'RXI_N.B.SI',
#        'RXI_N.B.SZ', 'RXI_N.B.TA', 'RXI_N.B.TH', 'RXI_N.B.VE', 'JRXWTFB_N.B',
#        'JRXWTFN_N.B', 'JRXWTFO_N.B', 'RXI$US_N.B.AL', 'RXI$US_N.B.NZ',
#        'RXI_N.B.CA', 'RXI_N.B.HK', 'RXI_N.B.SL']

# # the following chunk will get if the rate increased as compare to previous day
# # and they only need to be run once
# ex_df['Time Period'] = ex_df['Time Period'].apply(lambda x: x[:-2] + '20' + x[-2:] if int(x[-2:]) < 20 else x[:-2] + '19' + x[-2:])
# ex_df['Date'] = pd.to_datetime(ex_df['Time Period'], format="%d/%m/%Y")
# ex_df = ex_df.drop('Time Period', axis=1)
# ex_df = ex_df[ex_df['Date'] >= datetime.date(1999, 2, 8)].reset_index(drop=True)
# ex_df = ex_df.replace(r'ND', np.nan, regex=True)
# ex_df = ex_df.dropna()
# ex_df.to_csv('./exchange_rates.csv', index=False)

# size = ex_df.shape[0]
# cols = list(ex_df.columns)
# cols.remove('Date')
# for col in cols:
#     output_list = []
#     for i, row in ex_df.iterrows():
#         cur = row[col]
#         if (i == 0):
#             output_list.append(np.nan)
#             prev = cur
#             continue
#         else:
#             if cur > prev:
#                 output_list.append(1)
#             else:
#                 output_list.append(0)

#         prev = cur
#     #end for
#     ex_df[col + '_Change'] = pd.Series(output_list)
# #end for
# ex_df.to_csv('./exchange_rates.csv', index=False)
# print(ex_df.columns)
# ['RXI$US_N.B.EU', 'RXI$US_N.B.UK', 'RXI_N.B.BZ', 'RXI_N.B.CH',
#        'RXI_N.B.DN', 'RXI_N.B.IN', 'RXI_N.B.JA', 'RXI_N.B.KO', 'RXI_N.B.MA',
#        'RXI_N.B.MX', 'RXI_N.B.NO', 'RXI_N.B.SD', 'RXI_N.B.SF', 'RXI_N.B.SI',
#        'RXI_N.B.SZ', 'RXI_N.B.TA', 'RXI_N.B.TH', 'RXI_N.B.VE', 'JRXWTFB_N.B',
#        'JRXWTFN_N.B', 'JRXWTFO_N.B', 'RXI$US_N.B.AL', 'RXI$US_N.B.NZ',
#        'RXI_N.B.CA', 'RXI_N.B.HK', 'RXI_N.B.SL', 'Date',
#        'RXI$US_N.B.EU_Change', 'RXI$US_N.B.UK_Change', 'RXI_N.B.BZ_Change',
#        'RXI_N.B.CH_Change', 'RXI_N.B.DN_Change', 'RXI_N.B.IN_Change',
#        'RXI_N.B.JA_Change', 'RXI_N.B.KO_Change', 'RXI_N.B.MA_Change',
#        'RXI_N.B.MX_Change', 'RXI_N.B.NO_Change', 'RXI_N.B.SD_Change',
#        'RXI_N.B.SF_Change', 'RXI_N.B.SI_Change', 'RXI_N.B.SZ_Change',
#        'RXI_N.B.TA_Change', 'RXI_N.B.TH_Change', 'RXI_N.B.VE_Change',
#        'JRXWTFB_N.B_Change', 'JRXWTFN_N.B_Change', 'JRXWTFO_N.B_Change',
#        'RXI$US_N.B.AL_Change', 'RXI$US_N.B.NZ_Change', 'RXI_N.B.CA_Change',
#        'RXI_N.B.HK_Change', 'RXI_N.B.SL_Change']
'''

# ########################## Output Processed Data
# print(new_df['Date'])
# # 2002-01-02 to 2015-12-31

new_df = new_df.drop('Date', axis=1)
new_df.to_csv('processed_data.csv', index=False)
print(new_df.columns)
