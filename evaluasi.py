import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
file_path = 'data_zscore.csv'  # Path menuju file CSV
data = pd.read_csv(file_path)

# Preprocessing: Convert categorical features to numerical using LabelEncoder
label_encoder_dict = {}
for col in data.select_dtypes(include=['object']).columns:
    label_encoder = LabelEncoder()
    data[col] = label_encoder.fit_transform(data[col])
    label_encoder_dict[col] = label_encoder

# Check if the 'Died' column is already numeric
if data['Died'].dtype == 'object':
    # Convert the target column 'Died' to binary (e.g., 'Dead' = 1, 'Alive' = 0)
    data['Died'] = label_encoder_dict['Died'].transform(data['Died'])

# Split the data into features (X) and target (y)
X = data.drop('Died', axis=1)
y = data['Died']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree classifier
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Function to input data from terminal for each feature
def input_data():
    input_dict = {}
    print("Masukkan nilai untuk prediksi berdasarkan kolom berikut:")

    for col in X.columns:
        if col in label_encoder_dict:  # If the column is categorical
            categories = label_encoder_dict[col].classes_  # Get the possible values (categories)
            value = input(f"{col} (pilihan: {', '.join(categories)}): ")

            # If the value is categorical, encode it
            if value in categories:
                value = label_encoder_dict[col].transform([value])[0]
            else:
                print(f"Nilai tidak valid untuk kolom {col}, silakan masukkan salah satu dari {categories}.")
                return None  # End if invalid input
        else:  # If the column is numerical
            try:
                value = float(input(f"{col}: "))
            except ValueError:
                print(f"Nilai tidak valid untuk kolom {col}, harus berupa angka.")
                return None  # End if invalid input
        input_dict[col] = value
    return input_dict

# Predict the outcome based on terminal input
def predict_outcome():
    input_values = input_data()
    if input_values is None:
        return "Prediksi gagal, masukkan input yang valid."
    
    input_df = pd.DataFrame([input_values])  # Convert dict to DataFrame
    prediction = clf.predict(input_df)
    predicted_class = label_encoder_dict['Died'].inverse_transform([prediction[0]])  # Convert binary back to class
    return predicted_class[0]

# Run the prediction
if __name__ == "__main__":
    result = predict_outcome()
    print(f"Hasil prediksi: {result}")
