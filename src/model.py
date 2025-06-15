import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data_preprocessing import load_and_preprocess_data

# Load and preprocess data
df, X, y, le_title, le_wishlist, scaler = load_and_preprocess_data(r'C:\Users\name\OneDrive - Benha University (Faculty Of Computers & Information Technolgy)\Desktop\Bootcamp test\dataset\dataset.csv')

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Save everything:
joblib.dump(rf, 'src/random_forest_model.pkl')
joblib.dump(scaler, 'src/scaler.pkl')
joblib.dump(le_title, 'src/le_title.pkl')
joblib.dump(le_wishlist, 'src/le_wishlist.pkl')

print("✅ Model and preprocessing objects saved successfully!")
