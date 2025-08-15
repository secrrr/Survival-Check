from flask import Flask, render_template, request
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Inisialisasi Flask
app = Flask(__name__)

# Load model dari file (bisa digantikan dengan model buatan Anda)
def load_model():
    # Dummy: Load dataset dan lakukan preprocessing dan pelatihan model
    file_path = 'data_zscore.csv'  # Path file CSV Anda
    data = pd.read_csv(file_path)
    
    label_encoder_dict = {}
    for col in data.select_dtypes(include=['object']).columns:
        label_encoder = LabelEncoder()
        data[col] = label_encoder.fit_transform(data[col])
        label_encoder_dict[col] = label_encoder

    # Split data
    X = data.drop('Died', axis=1)
    y = data['Died']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Buat model Decision Tree
    clf = DecisionTreeClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    return clf, label_encoder_dict, X.columns

# Load model dan encoder saat aplikasi dijalankan
clf, label_encoder_dict, feature_columns = load_model()

# Route untuk halaman utama
@app.route('/')
def index():
    # Siapkan data dropdown untuk nilai kategorikal
    dropdown_data = {}
    for col in feature_columns:
        if col in label_encoder_dict:
            # Jika kolom kategorikal, tambahkan ke dropdown_data
            dropdown_data[col] = label_encoder_dict[col].inverse_transform(range(len(label_encoder_dict[col].classes_)))

    # Menyiapkan penjelasan untuk setiap kolom
    explanations = {
        'Age': 'Usia pasien dalam tahun',
        'Gender': 'Jenis kelamin pasien',
        'Headache': 'Riwayat sakit kepala yang dialami pasien',
        'Epilepsy': 'Kejang atau epilepsi pada pasien',
        'Hemparesis': 'Kelemahan otot sebagian sisi tubuh',
        'increaseICT': 'Tanda-tanda peningkatan tekanan intrakranial',
        'Pathology': 'Jenis diagnosis patologi tumor otak',
        'Pathology_Grade': 'Tingkat keparahan tumor, angka lebih tinggi menunjukkan tingkat yang lebih ganas (3/4)',
        'Thalamic_extension': 'Penyebaran tumor ke area thalamus',
        'Bil_extension': 'Apakah tumor menyebar ke kedua belahan otak',
        'Post_extension': ' Ekstensi atau penyebaran tumor ke bagian belakang otak',
        'BrainStem_extension': 'Ekstensi tumor ke batang otak',
        'MultiFocality': 'Adanya lebih dari satu titik tumor',
        'Midlineshift': 'Adanya pergeseran garis tengah otak yang dapat memengaruhi fungsi otak ',
        'Edema': 'Tingkat edema atau pembengkakan di sekitar tumor',
        'Approx_Tumor_Vol': 'Volume perkiraan tumor',
        'ExtentofSurgicalresection': 'Luas operasi yang telah dilakukan',
        'Shunt': 'Pemasangan shunt untuk mengalirkan cairan berlebih di otak',
        'ResidualsizeonMRI': 'Ukuran sisa tumor berdasarkan MRI setelah operasi',
        'Neurostate': 'Kondisi neurologis umum.',
        'PSBeforeRT': 'Skor performa sebelum radioterapi (angka lebih tinggi menunjukkan performa yang lebih baik)'
    }
    
    return render_template('index.html', columns=feature_columns, dropdown_data=dropdown_data, explanations=explanations)

# Route untuk melakukan prediksi
@app.route('/predict', methods=['POST'])
def predict():
    # Ambil data dari form
    input_data = {}
    for col in feature_columns:
        value = request.form.get(col)
        if col in label_encoder_dict:  # Jika kolom kategorikal
            value = label_encoder_dict[col].transform([value])[0]
        input_data[col] = float(value)
    
    # Buat DataFrame dari input
    input_df = pd.DataFrame([input_data])
    
    # Lakukan prediksi
    prediction = clf.predict(input_df)
    predicted_class = label_encoder_dict['Died'].inverse_transform([prediction[0]])
    
    # Tampilkan hasil prediksi
    return render_template('result.html', prediction=predicted_class[0])

if __name__ == '__main__':
    app.run(debug=True)
