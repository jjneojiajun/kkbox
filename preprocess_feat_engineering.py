import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count

train = pd.read_csv('./data/train.csv')
train = pd.concat((train, pd.read_csv('./data/train_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
test = pd.read_csv('./data/sample_submission_v2.csv')

members = pd.read_csv('./data/members_v3.csv')

transactions = pd.read_csv('./data/transactions.csv')
transactions = pd.concat((transactions, pd.read_csv('./data/transactions_v2.csv')), axis=0, ignore_index=True).reset_index(drop=True)
transactions = transactions.sort_values(by=['transaction_date'], ascending=[False]).reset_index(drop=True)
transactions = transactions.drop_duplicates(subset=['msno'], keep='first')

# Feature Engineering 
transactions['discount'] = transactions['plan_list_price'] - transactions['actual_amount_paid']
transactions['amt_per_day'] = transactions['actual_amount_paid'] / transactions['payment_plan_days']
transactions['is_discount'] = transactions.discount.apply(lambda x: 1 if x > 0 else 0)
transactions['membership_days'] = pd.to_datetime(transactions['membership_expire_date']).subtract(pd.to_datetime(transactions['transaction_date'])).dt.days.astype(int)

train['is_train'] = 1
test['is_train'] = 0
combined = pd.concat([train, test], axis=0)

combined = pd.merge(combined, members, how='left', on='msno')
members = []; print('members merge...')

gender = {'male':1, 'female':2}
combined['gender'] = combined['gender'].map(gender)

combined = pd.merge(combined, transactions, how='left', on='msno')
transactions =[]; print('transaction merge...')

train = combined[combined['is_train'] == 1]
test = combined[combined['is_train']  == 0]

train.drop(['is_train'], axis=1, inplace = True)
test.drop(['is_train'], axis=1, inplace = True)

del combined

def transform_df(df):
    df = pd.DataFrame(df)
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df

def transform_df2(df):
    df = df.sort_values(by=['date'], ascending=[False])
    df = df.reset_index(drop=True)
    df = df.drop_duplicates(subset=['msno'], keep='first')
    return df
last_user_logs = []

df_iter = pd.read_csv('./data/user_logs.csv', low_memory=False, iterator=True, chunksize=10000000)


i = 0 #~400 Million Records - starting at the end but remove locally if needed
for df in df_iter:
    if i>35:
        if len(df)>0:
            print(df.shape)
            p = Pool(cpu_count())
            df = p.map(transform_df, np.array_split(df, cpu_count()))   
            df = pd.concat(df, axis=0, ignore_index=True).reset_index(drop=True)
            df = transform_df2(df)
            p.close(); p.join()
            last_user_logs.append(df)
            print('...', df.shape)
            df = []
    i+=1


last_user_logs.append(transform_df(pd.read_csv('./data/user_logs_v2.csv')))
last_user_logs = pd.concat(last_user_logs, axis=0, ignore_index=True).reset_index(drop=True)
last_user_logs = transform_df2(last_user_logs)

train = pd.merge(train, last_user_logs, how='left', on='msno')
test = pd.merge(test, last_user_logs, how='left', on='msno')
last_user_logs=[]

train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)
test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)

train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)
test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)

train = train.fillna(0)
test = test.fillna(0)

train.to_csv('sorted_train.csv', encoding='utf-8', index= True)
test.to_csv('sorted_test.csv', encoding='utf-8', index= True)
