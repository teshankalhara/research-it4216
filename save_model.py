"""
save_model.py - Save your trained Random Forest model
Run this in your research environment where you have the trained model
"""

import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create model directory
os.makedirs("pdf-tampering-app/backend/model", exist_ok=True)

# ============================================
# METHOD 1: If you have your trained model from step4_working.py
# ============================================

# Load your features (from your research)
df_font = pd.read_csv("features/font_features.csv")
df_glyph = pd.read_csv("features/glyph_features.csv")

# Merge features (same as your step4_working.py)
df = pd.merge(df_font, df_glyph, on=['filename', 'label', 'tampering_type', 'doc_type'])

# Select features (same as your working model)
feature_cols = [
    'dominant_font_ratio', 'unique_fonts', 'font_variance', 
    'tampering_score_prelim', 'avg_space_ratio', 'font_size_variance',
    'avg_line_length', 'avg_word_length', 'space_ratio_std'
]

# Prepare data
X = df[feature_cols].fillna(0)
y = (df['label'] == 'tampered').astype(int)

# Train the model (same configuration as your 82.8% accuracy model)
print("Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
rf_model.fit(X, y)

# Train scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Model trained on {len(X)} samples")
print(f"Features used: {feature_cols}")
print(f"Feature importance: {rf_model.feature_importances_}")

# ============================================
# Save the model and scaler
# ============================================

# Save model
with open("pdf-tampering-app/backend/model/random_forest_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)
print("✓ Model saved: random_forest_model.pkl")

# Save scaler
with open("pdf-tampering-app/backend/model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("✓ Scaler saved: scaler.pkl")

# Save feature columns
with open("pdf-tampering-app/backend/model/feature_columns.pkl", "wb") as f:
    pickle.dump(feature_cols, f)
print("✓ Feature columns saved")

print("\n✅ Model export complete! Files saved to: pdf-tampering-app/backend/model/")