"""
train_model.py - Train and save model using your existing feature files
Place this in your backend folder and run once
"""

import pandas as pd
import pickle
import os
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

# Create model directory
os.makedirs("model", exist_ok=True)

# Path to your feature files (adjust paths as needed)
# Option 1: If features are in current directory
FONT_FEATURES_PATH = "../../features/font_features.csv"
GLYPH_FEATURES_PATH = "../../features/glyph_features.csv"

# Option 2: If features are in different location
# FONT_FEATURES_PATH = r"C:\Users\Disitha\Desktop\teshan research\pdf-tempering\features\font_features.csv"
# GLYPH_FEATURES_PATH = r"C:\Users\Disitha\Desktop\teshan research\pdf-tempering\features\glyph_features.csv"

print("="*60)
print("Training PDF Tampering Detection Model")
print("="*60)

try:
    # Load features
    df_font = pd.read_csv(FONT_FEATURES_PATH)
    df_glyph = pd.read_csv(GLYPH_FEATURES_PATH)
    print(f"✓ Loaded font features: {len(df_font)} records")
    print(f"✓ Loaded glyph features: {len(df_glyph)} records")
    
    # Merge
    df = pd.merge(df_font, df_glyph, on=['filename', 'label', 'tampering_type', 'doc_type'])
    print(f"✓ Merged dataset: {len(df)} records")
    
    # Feature columns (matching your research)
    feature_cols = [
        'dominant_font_ratio', 'unique_fonts', 'font_variance', 
        'tampering_score_prelim', 'avg_space_ratio', 'font_size_variance',
        'avg_line_length', 'avg_word_length', 'space_ratio_std'
    ]
    
    # Check which columns exist
    available_cols = [col for col in feature_cols if col in df.columns]
    print(f"✓ Using features: {available_cols}")
    
    # Prepare data
    X = df[available_cols].fillna(0)
    y = (df['label'] == 'tampered').astype(int)
    
    print(f"✓ Training samples: {len(X)}")
    print(f"✓ Genuine samples: {(y==0).sum()}")
    print(f"✓ Tampered samples: {(y==1).sum()}")
    
    # Train model
    print("\nTraining Random Forest...")
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )
    rf_model.fit(X, y)
    
    # Train scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Calculate accuracy on training data
    y_pred = rf_model.predict(X)
    accuracy = (y_pred == y).mean()
    print(f"✓ Training accuracy: {accuracy*100:.2f}%")
    
    # Feature importance
    print("\n📊 Feature Importance:")
    importance_df = pd.DataFrame({
        'feature': available_cols,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    for _, row in importance_df.iterrows():
        print(f"   {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    with open("model/random_forest_model.pkl", "wb") as f:
        pickle.dump(rf_model, f)
    print("\n✓ Model saved: model/random_forest_model.pkl")
    
    # Save scaler
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("✓ Scaler saved: model/scaler.pkl")
    
    # Save feature columns
    with open("model/feature_columns.pkl", "wb") as f:
        pickle.dump(available_cols, f)
    print("✓ Feature columns saved: model/feature_columns.pkl")
    
    print("\n✅ Model training complete!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\nPlease check that feature files exist at the specified paths.")