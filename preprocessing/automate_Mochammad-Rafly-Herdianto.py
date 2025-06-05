import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

# Fungsi-fungsi untuk preprocessing data

def load_data(file_path):
    """
    Memuat dataset dari file CSV.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset berhasil dimuat dari {file_path}")
        return df
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di path '{file_path}'. Pastikan path sudah benar.")
        return None
    except Exception as e:
        print(f"Error lain saat memuat data: {e}")
        return None

def convert_date_column(df, column_name='Date'):
    """
    Mengonversi kolom tanggal ke tipe datetime.
    """
    if column_name in df.columns:
        try:
            df[column_name] = pd.to_datetime(df[column_name])
            print(f"Kolom '{column_name}' berhasil dikonversi ke datetime.")
        except Exception as e:
            print(f"Error saat mengkonversi kolom '{column_name}': {e}")
    else:
        print(f"Kolom '{column_name}' tidak ditemukan untuk konversi tanggal.")
    return df

def handle_duplicates(df):
    """
    Menangani data duplikat (berdasarkan EDA, tidak ada yang diharapkan).
    """
    initial_rows = len(df)
    df.drop_duplicates(inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"{rows_dropped} baris duplikat telah dihapus.")
        df.reset_index(drop=True, inplace=True)
    else:
        print("Tidak ada baris duplikat untuk dihapus (sesuai EDA).")
    return df

def engineer_date_features(df, date_column='Date'):
    """
    Membuat fitur baru dari kolom tanggal.
    """
    if date_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[date_column]):
        df['Year'] = df[date_column].dt.year
        df['Month'] = df[date_column].dt.month
        df['Day'] = df[date_column].dt.day
        df['DayOfWeek'] = df[date_column].dt.dayofweek # Senin=0, Minggu=6
        df['WeekOfYear'] = df[date_column].dt.isocalendar().week.astype(int)
        print("Fitur tanggal (Year, Month, Day, DayOfWeek, WeekOfYear) berhasil dibuat.")
    else:
        print(f"Kolom '{date_column}' bukan datetime atau tidak ditemukan. Lewati feature engineering tanggal.")
    return df

def handle_outliers_iqr_clip(df, columns_to_treat):
    """
    Menangani outlier menggunakan metode IQR clipping.
    """
    print("\nMenangani Outlier (IQR Clipping):")
    for col in columns_to_treat:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
            if outliers_before > 0:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                print(f"  Outlier di '{col}' telah di-clip ke rentang [{lower_bound:.2f}, {upper_bound:.2f}]. {outliers_before} nilai terpengaruh.")
            else:
                print(f"  Tidak ada outlier signifikan (berdasarkan IQR 1.5x) di '{col}' yang memerlukan clipping.")
        else:
            print(f"  Kolom '{col}' untuk penanganan outlier tidak ditemukan atau bukan numerik, dilewati.")
    return df

def encode_categorical_features(df, drop_date_column='Date'):
    """
    Melakukan One-Hot Encoding pada fitur kategorikal dan menghapus kolom tanggal asli jika perlu.
    """
    categorical_cols_to_encode = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    if not categorical_cols_to_encode:
        print("Tidak ada kolom kategorikal untuk di-encode.")
    else:
        df = pd.get_dummies(df, columns=categorical_cols_to_encode, drop_first=True)
        print(f"Kolom kategorikal berikut telah di-OneHotEncode (drop_first=True): {categorical_cols_to_encode}")

    # Menghapus kolom 'Date' asli setelah feature engineering, jika masih ada dan bertipe datetime
    if drop_date_column in df.columns and pd.api.types.is_datetime64_any_dtype(df[drop_date_column]):
        df = df.drop(columns=[drop_date_column])
        print(f"Kolom '{drop_date_column}' asli (datetime) telah dihapus.")
    elif drop_date_column in df.columns:
        print(f"Kolom '{drop_date_column}' masih ada tetapi bukan tipe datetime. Tidak dihapus oleh fungsi ini.")

    return df

def scale_numerical_features(df):
    """
    Melakukan scaling pada fitur numerik menggunakan StandardScaler.
    """
    # Memastikan hanya kolom numerik yang di-scale
    numerical_features = df.select_dtypes(include=np.number).columns.tolist()
    
    if not numerical_features:
        print("Tidak ada fitur numerik untuk di-scale.")
        return df
        
    scaler = StandardScaler()
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    print("Semua fitur numerik telah di-scale menggunakan StandardScaler.")
    return df

def preprocess_data(raw_data_path, output_file_path):
    """
    Fungsi utama untuk menjalankan semua langkah preprocessing.
    """
    print("Memulai proses preprocessing data otomatis...")
    
    # 1. Data Loading
    df = load_data(raw_data_path)
    if df is None:
        return None
    
    initial_shape = df.shape
    print(f"Dimensi data awal: {initial_shape}")

    # 2. Konversi Tipe Data Tanggal
    df = convert_date_column(df, column_name='Date')
    
    # 3. Penanganan Nilai Hilang 
    missing_values_count = df.isnull().sum().sum()
    if missing_values_count > 0:
        print(f"PERINGATAN: Ditemukan {missing_values_count} nilai hilang. Anda mungkin perlu menambahkan langkah penanganan.")
    else:
        print("Tidak ada nilai hilang yang terdeteksi (sesuai EDA).")

    # 4. Penanganan Data Duplikat 
    df = handle_duplicates(df)
    
    # 5. Feature Engineering dari Tanggal
    df = engineer_date_features(df, date_column='Date')
    
    # 6. Penanganan Outlier (sesuai EDA untuk 'Units Sold', 'Units Returned')
    cols_for_outlier_treatment = ['Units Sold', 'Units Returned']
    df = handle_outliers_iqr_clip(df, columns_to_treat=cols_for_outlier_treatment)
    
    # 7. Encoding Fitur Kategorikal (dan hapus kolom 'Date' asli jika sudah diproses)
    df = encode_categorical_features(df, drop_date_column='Date')
    
    # 8. Feature Scaling
    df = scale_numerical_features(df)
    
    final_shape = df.shape
    print(f"\nPreprocessing data otomatis selesai.")
    print(f"Dimensi data setelah preprocessing: {final_shape}")
    
    # Menyimpan dataset yang sudah diproses
    try:
        output_dir = os.path.dirname(output_file_path)
        if output_dir and not os.path.exists(output_dir): 
            os.makedirs(output_dir)
            
        df.to_csv(output_file_path, index=False)
        print(f"Dataset yang sudah diproses berhasil disimpan ke: {os.path.abspath(output_file_path)}")
    except Exception as e:
        print(f"Error saat menyimpan dataset yang sudah diproses: {e}")
        return None 
        
    return df

if __name__ == "__main__":
    # Path ke dataset mentah.
    raw_dataset_path = '../supplement_sales_raw.csv'
    
    # Path untuk menyimpan dataset yang sudah diproses.
    preprocessed_output_file = 'supplement_sales_preprocessing.csv'

    print(f"Lokasi direktori kerja saat skrip dijalankan: {os.getcwd()}")
    print(f"Path data mentah yang akan diakses: {os.path.abspath(raw_dataset_path)}")
    print(f"Path data terproses yang akan disimpan: {os.path.abspath(preprocessed_output_file)}")

    # Menjalankan fungsi preprocessing utama
    df_preprocessed_script = preprocess_data(raw_data_path=raw_dataset_path, 
                                             output_file_path=preprocessed_output_file)
    
    if df_preprocessed_script is not None:
        print("\nContoh data yang telah diproses oleh skrip (5 baris pertama):")
        print(df_preprocessed_script.head())
        
        # Verifikasi apakah file output benar-benar ada
        if os.path.exists(preprocessed_output_file):
            print(f"\nVERIFIKASI: File '{preprocessed_output_file}' berhasil dibuat di '{os.path.abspath(preprocessed_output_file)}'.")
        else:
            print(f"\nPERHATIAN: File '{preprocessed_output_file}' TIDAK ditemukan setelah skrip dijalankan.")
    else:
        print("\nProses preprocessing tidak menghasilkan DataFrame atau gagal.")