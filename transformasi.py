import pandas as pd
from scipy import stats
import pandasgui as pg

# Mengimpor dataset bersih dari file CSV baru
data = pd.read_csv('data_clean.csv')

# Menghitung Z-score hanya untuk kolom 'Approx_Tumor_Vol'
z_score = data.copy()
z_score['Approx_Tumor_Vol'] = (z_score['Approx_Tumor_Vol'] - z_score['Approx_Tumor_Vol'].mean()) / z_score['Approx_Tumor_Vol'].std()

# Menyimpan dataset dengan hasil Z-score pada 'Approx_Tumor_Vol' ke file CSV baru
# data.to_csv('data_zscore.csv', index=False)

print("Transformasi Z-score pada 'Approx_Tumor_Vol' selesai dan disimpan ke 'data_zscore.csv'")

# Menampilkan data asli dan Z-score di Pandas GUI
gui = pg.show(data, z_score)
