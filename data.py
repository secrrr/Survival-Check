import numpy as np
import pandas as pd

# Importing the dataset
data = pd.read_csv('data.csv')
print(data)

# Cek missing value
print("Missing Value:")
print(data.isna().sum())

# Inisialisasi dictionary untuk menyimpan outlier
outliers_dict = {}
outliers_count = {}
data_clean = data.copy()

# Loop untuk mendeteksi outliers di setiap kolom numerik
for column in data.select_dtypes(include=[np.number]).columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Menentukan batas bawah dan batas atas untuk outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Mengidentifikasi outliers
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    
    # Simpan hasil outliers di dictionary
    outliers_dict[column] = outliers

    # Hapus outliers dari data_clean
    data_clean = data_clean[(data_clean[column] >= lower_bound) & (data_clean[column] <= upper_bound)]
    
    # Hitung jumlah outliers dan simpan
    outliers_count[column] = len(outliers)

# Print jumlah outliers di setiap kolom
print("\nJumlah outliers di setiap kolom:")
for column, count in outliers_count.items():
    print(f"{column}: {count} outliers")

# Print outliers pada kolom 'Approx_Tumor_Vol'
print("\nOutliers pada kolom 'Approx_Tumor_Vol':")
print(outliers_dict['Approx_Tumor_Vol'][['Approx_Tumor_Vol']])

# Menyimpan dataset baru tanpa outliers ke file CSV
# data_clean.to_csv('data_clean.csv', index=False)

import pandasgui as pg
gui = pg.show(data, data_clean, outliers_dict['Approx_Tumor_Vol'][['Approx_Tumor_Vol']])