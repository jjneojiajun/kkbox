import numpy as np
import pandas as pd

from multiprocessing import Pool, cpu_count

# fixing dtypes: time and numeric variables
def fix_dtypes(df, time_cols, num_cols):
    
    print('***************************************************************')
    print('Begin fixing data types: ')
    print('***************************************************************\n')
    
    def fix_time_col(df, time_cols):
        for time_col in time_cols:
            df[time_col] = pd.to_datetime(df[time_col], errors = 'coerce', format = '%Y%m%d')
        print('---------------------------------------------------------------')
        print('The following time columns has been fixed: ')
        print(time_cols)
        print('---------------------------------------------------------------\n')

    def fix_num_col(df, num_cols):
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors = 'coerce')
        print('---------------------------------------------------------------')
        print('The following number columns has been fixed: ')
        print(num_cols)
        print('---------------------------------------------------------------\n')
        
    if(len(num_cols) > 0):
        fix_num_col(df, num_cols)
    fix_time_col(df, time_cols)

    print('---------------------------------------------------------------')
    print('Final data types:')
    result = pd.DataFrame(df.dtypes, columns = ['type'])
    result = result.reindex(result['type'].astype(str).str.len().sort_values().index)
    print(result)
    print('_______________________________________________________________\n\n\n')
    return df

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
del last_user_logs

train = train.rename(columns = {'date':'user_log_date'})
test = test.rename(columns = {'date':'user_log_date'})

train['user_log_date'] = train['user_log_date'].fillna(20170105.0)
train = fix_dtypes(train, time_cols = ['transaction_date', 'membership_expire_date', 'registration_init_time', 'user_log_date'], num_cols = [])

test['user_log_date'] = test['user_log_date'].fillna(20170105.0)
test = fix_dtypes(test, time_cols = ['transaction_date', 'membership_expire_date', 'registration_init_time', 'user_log_date'], num_cols = [])

# Splitting of Dates into Days and Months Respectively

date_dict = {'t_':'transaction_date', 'm_':'membership_expire_date', \
             'r_':'registration_init_time', 'l_':'user_log_date'}

print("Creating date columns... ")

for key in date_dict:  
    if key == 'r_':
        train[key+'month'] = [d.month for d in train[date_dict[key]]]
        train[key+'day'] = [d.day for d in train[date_dict[key]]]
    else:
        train[key+'day'] = [d.day for d in train[date_dict[key]]]

train['transaction_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in train['transaction_date']]
train['membership_expire_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in train['membership_expire_date']]
train['registration_init_time'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in train['registration_init_time']]
train['last_user_log_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in train['last_user_log_date']]
print("Done!")

print("Creating date columns for test... ")
for key in date_dict:  
    if key == 'r_':
        test[key+'month'] = [d.month for d in test[date_dict[key]]]
        test[key+'day'] = [d.day for d in test[date_dict[key]]]
    else:
        test[key+'day'] = [d.day for d in test[date_dict[key]]]

test['transaction_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in test['transaction_date']]
test['membership_expire_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in test['membership_expire_date']]
test['registration_init_time'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in test['registration_init_time']]
test['last_user_log_date'] = [d.year + (d.month-1) / 12 + d.day / 365 for d in test['last_user_log_date']]
print("Done!")

# get the feature of whether user automatically renew and not cancel and vice versa

train['autorenew_&_not_cancel'] = ((train.is_auto_renew == 1) == (train.is_cancel == 0)).astype(np.int8)
test['autorenew_&_not_cancel'] = ((test.is_auto_renew == 1) == (test.is_cancel == 0)).astype(np.int8)

train['notAutorenew_&_cancel'] = ((train.is_auto_renew == 0) == (train.is_cancel == 1)).astype(np.int8)
test['notAutorenew_&_cancel'] = ((test.is_auto_renew == 0) == (test.is_cancel == 1)).astype(np.int8)

train = train.fillna(0)
test = test.fillna(0)

train.to_csv('sorted_train_v3.csv', encoding='utf-8', index= True)
test.to_csv('sorted_test_v3.csv', encoding='utf-8', index= True)
