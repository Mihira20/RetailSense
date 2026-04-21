import pandas as pd
import numpy as np
import os

def load_raw_data(data_dir='data/raw'):
    """Load all 4 raw CSV files"""
    train = pd.read_csv(f'{data_dir}/train.csv')
    stores = pd.read_csv(f'{data_dir}/stores.csv')
    features = pd.read_csv(f'{data_dir}/features.csv')
    test = pd.read_csv(f'{data_dir}/test.csv')
    print(f"✅ Loaded: train{train.shape}, stores{stores.shape}, features{features.shape}, test{test.shape}")
    return train, stores, features, test

def merge_data(train, stores, features):
    """Merge all datasets into one clean dataframe"""
    df = train.merge(stores, on='Store', how='left')
    df = df.merge(features, on=['Store', 'Date', 'IsHoliday'], how='left')
    print(f"✅ Merged shape: {df.shape}")
    return df

def clean_data(df):
    """Handle missing values and data types"""
    # Convert date
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Fill MarkDowns with 0
    markdown_cols = ['MarkDown1','MarkDown2','MarkDown3','MarkDown4','MarkDown5']
    df[markdown_cols] = df[markdown_cols].fillna(0)
    
    # Forward fill CPI and Unemployment
    df['CPI'] = df['CPI'].ffill()
    df['Unemployment'] = df['Unemployment'].ffill()
    
    # Remove negative sales
    df = df[df['Weekly_Sales'] >= 0].copy()
    
    print(f"✅ Cleaned shape: {df.shape}")
    print(f"✅ Missing values: {df.isnull().sum().sum()}")
    return df

def add_time_features(df):
    """Extract time-based features"""
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    
    # Is it a holiday season? (Nov-Dec)
    df['IsHolidaySeason'] = df['Month'].isin([11, 12]).astype(int)
    
    print("✅ Time features added")
    return df

def save_processed(df, output_dir='data/processed'):
    """Save processed dataframe"""
    os.makedirs(output_dir, exist_ok=True)
    output_path = f'{output_dir}/merged_clean.csv'
    df.to_csv(output_path, index=False)
    print(f"✅ Saved to {output_path}")

def run_pipeline(data_dir='data/raw'):
    """Run the full pipeline"""
    train, stores, features, test = load_raw_data(data_dir)
    df = merge_data(train, stores, features)
    df = clean_data(df)
    df = add_time_features(df)
    save_processed(df)
    return df

if __name__ == '__main__':
    df = run_pipeline()
    print("\n✅ Pipeline complete!")
    print(df.head())