import pandas as pd

def numerize_csv(path: str):
    ''' Takes a path to a project csv and converts its entries to numerical '''
    df = pd.read_csv(path)
    df['gender'] = (df['gender'] == 'Female').astype(int)

    for header in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', \
                'Discontinued']:
        df[header] = (df[header] == 'Yes').astype(int)

    for header in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', \
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        # some here have the value no phone/internet service, which are casted to 0
        df[header] = (df[header] == 'Yes').astype(int)

    for header in ['MonthlyCharges', 'TotalCharges', 'tenure']:
        # lines that need normalization
        df[header] /= (max(df[header]) - min(df[header]))

    df['InternetService'] = df['InternetService'].map({'Fiber optic': 2, 'DSL': 1, 'No': 0})
    df['Contract'] = df['Contract'].map({'Two year': 2, 'One year': 1, 'Month-to-month': 0})
    # Note that the PaymentMethod column contains some entries that are marked automatic
    # that's probably correlated with discontinuation in some way.
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Credit card (automatic)': 3,
        'Electronic check': 2,
        'Bank transfer (automatic)': 1,
        'Mailed check': 0})
    df.drop('customerID', axis=1, inplace=True)
    mean = df.mean()
    df.fillna(mean, inplace=True)
    return df

def numerize_csv_test(path: str):
    ''' Takes a path to a project csv and converts its entries to numerical '''
    df = pd.read_csv(path)
    df['gender'] = (df['gender'] == 'Female').astype(int)

    for header in ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']:
        df[header] = (df[header] == 'Yes').astype(int)

    for header in ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', \
                'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']:
        # some here have the value no phone/internet service, which are casted to 0
        df[header] = (df[header] == 'Yes').astype(int)

    for header in ['MonthlyCharges', 'TotalCharges', 'tenure']:
        # lines that need normalization
        df[header] /= (max(df[header]) - min(df[header]))

    df['InternetService'] = df['InternetService'].map({'Fiber optic': 2, 'DSL': 1, 'No': 0})
    df['Contract'] = df['Contract'].map({'Two year': 2, 'One year': 1, 'Month-to-month': 0})
    # Note that the PaymentMethod column contains some entries that are marked automatic
    # that's probably correlated with discontinuation in some way.
    df['PaymentMethod'] = df['PaymentMethod'].map({
        'Credit card (automatic)': 3,
        'Electronic check': 2,
        'Bank transfer (automatic)': 1,
        'Mailed check': 0})
    df.drop('customerID', axis=1, inplace=True)
    mean = df.mean()
    df.fillna(mean, inplace=True)
    return df

def combine_related_columns(df: pd.DataFrame):
    ''' Takes a project dataframe and combines its related rows. '''
    df_cpy = df
    PHONE_SERVICE_WEIGHT = 0.7
    TV_STREAM_WEIGHT = 0.5
    SECURITY_WEIGHTS = {
        'security': 0.25,
        'backup': 0.25,
        'protection': 0.25,
        'support': 0.25
    }
    df_cpy['PhoneUsageScore'] = df_cpy.pop('PhoneService').values * PHONE_SERVICE_WEIGHT \
                                + df_cpy.pop('MultipleLines').values * (1 - PHONE_SERVICE_WEIGHT)
    df_cpy['InternetSecurityScore'] = df_cpy.pop('OnlineSecurity').values * SECURITY_WEIGHTS.get('security') \
                                        + df_cpy.pop('OnlineBackup').values * SECURITY_WEIGHTS.get('backup') \
                                        + df_cpy.pop('DeviceProtection').values * SECURITY_WEIGHTS.get('protection') \
                                        + df_cpy.pop('TechSupport').values * SECURITY_WEIGHTS.get('support')
    df_cpy['InternetStreamingScore'] = df_cpy.pop('StreamingTV').values * TV_STREAM_WEIGHT \
                                        + df_cpy.pop('StreamingMovies').values * (1 - TV_STREAM_WEIGHT)
    df_cpy.insert(0, 'Discontinued', df_cpy.pop('Discontinued'))
    return df_cpy

def combine_related_columns_test(df: pd.DataFrame):
    ''' Takes a project dataframe and combines its related rows. '''
    df_cpy = df
    PHONE_SERVICE_WEIGHT = 0.7
    TV_STREAM_WEIGHT = 0.5
    SECURITY_WEIGHTS = {
        'security': 0.25,
        'backup': 0.25,
        'protection': 0.25,
        'support': 0.25
    }
    df_cpy['PhoneUsageScore'] = df_cpy.pop('PhoneService').values * PHONE_SERVICE_WEIGHT \
                                + df_cpy.pop('MultipleLines').values * (1 - PHONE_SERVICE_WEIGHT)
    df_cpy['InternetSecurityScore'] = df_cpy.pop('OnlineSecurity').values * SECURITY_WEIGHTS.get('security') \
                                        + df_cpy.pop('OnlineBackup').values * SECURITY_WEIGHTS.get('backup') \
                                        + df_cpy.pop('DeviceProtection').values * SECURITY_WEIGHTS.get('protection') \
                                        + df_cpy.pop('TechSupport').values * SECURITY_WEIGHTS.get('support')
    df_cpy['InternetStreamingScore'] = df_cpy.pop('StreamingTV').values * TV_STREAM_WEIGHT \
                                        + df_cpy.pop('StreamingMovies').values * (1 - TV_STREAM_WEIGHT)
    return df_cpy

def write_submission(preds):
    ''' Utility function to write to the submission file. '''
    df = pd.read_csv('submission.csv')
    pred_len = len(preds)
    target_len = len(df['ID'])
    if pred_len != target_len:
        raise ValueError(f'Incorrect input length. Required: {target_len}. Provided: {pred_len}')
    df['TARGET'] = pd.Series(preds)
    df.set_index('ID', inplace=True)
    df.to_csv('submission.csv')