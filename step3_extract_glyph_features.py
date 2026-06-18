"""
STEP 3 FIXED: Extract Visual Glyph Features from PDF Text
Handles variable PDF sizes and proper image conversion
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import cv2
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "dataset"
FEATURES_PATH = "features/font_features.csv"
OUTPUT_PATH = "features/glyph_features.csv"

def extract_glyph_features_from_pdf(pdf_path):
    """
    Extract visual glyph features from PDF pages - FIXED VERSION
    """
    try:
        doc = fitz.open(pdf_path)
        
        glyph_features = {
            'avg_stroke_width': [],
            'text_density': [],
            'spacing_variance': [],
            'sharpness_scores': [],
            'contrast_scores': []
        }
        
        # Analyze first 3 pages or all pages if fewer
        num_pages = min(len(doc), 3)
        
        for page_num in range(num_pages):
            page = doc[page_num]
            
            # Render page to image with fixed DPI
            zoom = 2  # 2x zoom for better detail
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            # Convert to numpy array correctly
            img = np.frombuffer(pix.tobraytes(), dtype=np.uint8).reshape(pix.height, pix.width, 3)
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            
            # 1. Extract stroke width using morphological operations
            # Threshold to get text
            _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours of text
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            stroke_widths = []
            for contour in contours:
                if len(contour) > 5:
                    # Calculate bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 5 and h > 5:  # Ignore very small contours
                        # Approximate stroke width as min dimension
                        stroke_width = min(w, h) / 3  # Rough estimate
                        if 0.5 < stroke_width < 15:
                            stroke_widths.append(stroke_width)
            
            avg_stroke = np.mean(stroke_widths) if stroke_widths else 0
            glyph_features['avg_stroke_width'].append(avg_stroke)
            
            # 2. Text density
            text_pixels = np.sum(binary > 0)
            total_pixels = binary.size
            text_density = text_pixels / total_pixels if total_pixels > 0 else 0
            glyph_features['text_density'].append(text_density)
            
            # 3. Spacing analysis (horizontal profile)
            horizontal_profile = np.sum(binary, axis=0) / 255
            # Find peaks (character positions)
            peaks = []
            for i in range(1, len(horizontal_profile) - 1):
                if horizontal_profile[i] > horizontal_profile[i-1] and horizontal_profile[i] > horizontal_profile[i+1]:
                    if horizontal_profile[i] > 10:  # Threshold
                        peaks.append(i)
            
            if len(peaks) > 1:
                spacing = np.diff(peaks)
                spacing_variance = np.var(spacing) if len(spacing) > 0 else 0
            else:
                spacing_variance = 0
            glyph_features['spacing_variance'].append(spacing_variance)
            
            # 4. Sharpness (using Laplacian variance)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = laplacian.var()
            glyph_features['sharpness_scores'].append(sharpness)
            
            # 5. Contrast (using standard deviation)
            contrast = np.std(gray)
            glyph_features['contrast_scores'].append(contrast)
        
        doc.close()
        
        # Aggregate features
        if len(glyph_features['avg_stroke_width']) > 0:
            return {
                'avg_stroke_width': np.mean(glyph_features['avg_stroke_width']),
                'stroke_width_std': np.std(glyph_features['avg_stroke_width']) if len(glyph_features['avg_stroke_width']) > 1 else 0,
                'avg_text_density': np.mean(glyph_features['text_density']),
                'text_density_std': np.std(glyph_features['text_density']) if len(glyph_features['text_density']) > 1 else 0,
                'avg_spacing_variance': np.mean(glyph_features['spacing_variance']),
                'spacing_variance_std': np.std(glyph_features['spacing_variance']) if len(glyph_features['spacing_variance']) > 1 else 0,
                'avg_sharpness': np.mean(glyph_features['sharpness_scores']),
                'avg_contrast': np.mean(glyph_features['contrast_scores']),
                'glyph_consistency_score': 1.0 - (np.std(glyph_features['sharpness_scores']) / (np.mean(glyph_features['sharpness_scores']) + 0.01))
            }
        else:
            return None
        
    except Exception as e:
        # Silent fail for individual PDF errors
        return None

def extract_all_glyph_features():
    """Extract glyph features from all PDFs"""
    
    print("="*60)
    print("STEP 3 FIXED: EXTRACTING VISUAL GLYPH FEATURES")
    print("Analyzing text shapes, stroke width, and spacing patterns...")
    print("="*60)
    
    # Load font features to get the PDF list
    if not os.path.exists(FEATURES_PATH):
        print(f"Error: Run STEP 2 first to generate {FEATURES_PATH}")
        return None
    
    df_font = pd.read_csv(FEATURES_PATH)
    print(f"✓ Loaded {len(df_font)} PDFs from font features")
    
    all_glyph_features = []
    failed_count = 0
    
    for idx, row in tqdm(df_font.iterrows(), total=len(df_font), desc="Extracting glyph features"):
        pdf_path = os.path.join(DATASET_PATH, row['filename'])
        
        if os.path.exists(pdf_path):
            glyph_features = extract_glyph_features_from_pdf(pdf_path)
            
            if glyph_features:
                glyph_features['filename'] = row['filename']
                glyph_features['label'] = row['label']
                glyph_features['tampering_type'] = row['tampering_type']
                glyph_features['doc_type'] = row['doc_type']
                all_glyph_features.append(glyph_features)
            else:
                failed_count += 1
        else:
            failed_count += 1
    
    # Save features
    if all_glyph_features:
        df_glyph = pd.DataFrame(all_glyph_features)
        df_glyph.to_csv(OUTPUT_PATH, index=False)
        
        print("\n" + "="*60)
        print("GLYPH FEATURE EXTRACTION SUMMARY")
        print("="*60)
        print(f"✓ Successfully processed: {len(df_glyph)} PDFs")
        print(f"⚠ Failed/Skipped: {failed_count} PDFs")
        print(f"✓ Features saved: {OUTPUT_PATH}")
        
        # Statistics by label
        if 'label' in df_glyph.columns:
            genuine = df_glyph[df_glyph['label'] == 'genuine']
            tampered = df_glyph[df_glyph['label'] == 'tampered']
            
            print(f"\n📊 Glyph Features Comparison:")
            print(f"\n   Genuine PDFs (n={len(genuine)}):")
            print(f"     - Avg Stroke Width: {genuine['avg_stroke_width'].mean():.2f}")
            print(f"     - Avg Text Density: {genuine['avg_text_density'].mean():.4f}")
            print(f"     - Avg Spacing Variance: {genuine['avg_spacing_variance'].mean():.2f}")
            print(f"     - Glyph Consistency: {genuine['glyph_consistency_score'].mean():.3f}")
            
            print(f"\n   Tampered PDFs (n={len(tampered)}):")
            print(f"     - Avg Stroke Width: {tampered['avg_stroke_width'].mean():.2f}")
            print(f"     - Avg Text Density: {tampered['avg_text_density'].mean():.4f}")
            print(f"     - Avg Spacing Variance: {tampered['avg_spacing_variance'].mean():.2f}")
            print(f"     - Glyph Consistency: {tampered['glyph_consistency_score'].mean():.3f}")
            
            # Key differences
            print(f"\n   🔍 KEY DIFFERENCES (Tampered - Genuine):")
            print(f"     Stroke Width: {tampered['avg_stroke_width'].mean() - genuine['avg_stroke_width'].mean():+.2f}")
            print(f"     Text Density: {tampered['avg_text_density'].mean() - genuine['avg_text_density'].mean():+.4f}")
            print(f"     Spacing Variance: {tampered['avg_spacing_variance'].mean() - genuine['avg_spacing_variance'].mean():+.2f}")
        
        return df_glyph
    else:
        print("\n❌ No glyph features were extracted successfully!")
        print("This might be due to PDF rendering issues.")
        print("Creating empty dataframe with necessary columns...")
        
        # Create empty dataframe with proper structure
        df_glyph = pd.DataFrame(columns=[
            'avg_stroke_width', 'stroke_width_std', 'avg_text_density', 
            'text_density_std', 'avg_spacing_variance', 'spacing_variance_std',
            'avg_sharpness', 'avg_contrast', 'glyph_consistency_score',
            'filename', 'label', 'tampering_type', 'doc_type'
        ])
        df_glyph.to_csv(OUTPUT_PATH, index=False)
        return df_glyph

def combine_features():
    """Combine font features and glyph features for final analysis"""
    
    print("\n" + "="*60)
    print("COMBINING FONT + GLYPH FEATURES")
    print("="*60)
    
    # Load both feature sets
    df_font = pd.read_csv("features/font_features.csv")
    df_glyph = pd.read_csv("features/glyph_features.csv")
    
    if len(df_glyph) == 0:
        print("⚠ No glyph features available. Using font features only.")
        df_combined = df_font.copy()
    else:
        # Merge on filename
        df_combined = pd.merge(df_font, df_glyph, on=['filename', 'label', 'tampering_type', 'doc_type'], how='left')
    
    # Fill NaN values with defaults
    df_combined = df_combined.fillna(0)
    
    # Calculate combined tampering score
    # Weighted combination of font and glyph indicators
    df_combined['combined_tampering_score'] = (
        df_combined['tampering_score_prelim'] * 0.5 +  # Font inconsistency
        (1 - df_combined['glyph_consistency_score'].fillna(1)) * 0.3 +  # Glyph inconsistency
        (df_combined['font_variance'].fillna(0)) * 0.2  # Font variance
    )
    
    # Save combined features
    df_combined.to_csv("features/combined_features.csv", index=False)
    print(f"✓ Combined features saved: features/combined_features.csv")
    print(f"✓ Total PDFs analyzed: {len(df_combined)}")
    
    return df_combined

def visualize_glyph_features(df_glyph):
    """Create visualizations of glyph features"""
    if len(df_glyph) == 0:
        print("No glyph data to visualize")
        return
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs("features/plots", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Stroke width comparison
        ax1 = axes[0, 0]
        for label, color in [('genuine', 'green'), ('tampered', 'red')]:
            data = df_glyph[df_glyph['label'] == label]['avg_stroke_width'].dropna()
            if len(data) > 0:
                ax1.hist(data, alpha=0.5, bins=30, label=label, color=color)
        ax1.set_xlabel('Average Stroke Width')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Stroke Width: Genuine vs Tampered')
        ax1.legend()
        
        # Plot 2: Text density comparison
        ax2 = axes[0, 1]
        for label, color in [('genuine', 'green'), ('tampered', 'red')]:
            data = df_glyph[df_glyph['label'] == label]['avg_text_density'].dropna()
            if len(data) > 0:
                ax2.hist(data, alpha=0.5, bins=30, label=label, color=color)
        ax2.set_xlabel('Average Text Density')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Text Density: Genuine vs Tampered')
        ax2.legend()
        
        # Plot 3: Glyph consistency boxplot
        ax3 = axes[1, 0]
        df_filtered = df_glyph[df_glyph['label'].isin(['genuine', 'tampered'])]
        if len(df_filtered) > 0:
            sns.boxplot(x='label', y='glyph_consistency_score', data=df_filtered, ax=ax3)
            ax3.set_xlabel('Document Type')
            ax3.set_ylabel('Glyph Consistency Score')
            ax3.set_title('Glyph Consistency: Genuine vs Tampered')
        
        # Plot 4: Spacing variance by tampering type
        ax4 = axes[1, 1]
        tampering_types = df_glyph[df_glyph['tampering_type'] != 'none']['tampering_type'].unique()
        if len(tampering_types) > 0:
            data_to_plot = []
            labels = []
            for t in tampering_types:
                data = df_glyph[df_glyph['tampering_type'] == t]['avg_spacing_variance'].dropna()
                if len(data) > 0:
                    data_to_plot.append(data)
                    labels.append(t)
            if data_to_plot:
                ax4.boxplot(data_to_plot, labels=labels)
                ax4.set_xlabel('Tampering Type')
                ax4.set_ylabel('Spacing Variance')
                ax4.set_title('Spacing Variance by Tampering Method')
                ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig("features/plots/glyph_analysis.png", dpi=150)
        print(f"✓ Visualization saved: features/plots/glyph_analysis.png")
        plt.close()
        
    except Exception as e:
        print(f"Visualization error (install matplotlib): {e}")

if __name__ == "__main__":
    # Extract glyph features
    df_glyph = extract_all_glyph_features()
    
    if df_glyph is not None:
        # Create visualizations
        visualize_glyph_features(df_glyph)
        
        # Combine features
        df_combined = combine_features()
        
        # Show sample
        print("\n" + "="*60)
        print("SAMPLE COMBINED FEATURES (First 10 PDFs)")
        print("="*60)
        display_cols = ['filename', 'label', 'tampering_type', 
                       'dominant_font_ratio', 'glyph_consistency_score', 
                       'combined_tampering_score']
        available_cols = [c for c in display_cols if c in df_combined.columns]
        print(df_combined[available_cols].head(10))
        
        print("\n" + "="*60)
        print("✅ STEP 3 COMPLETE!")
        print("="*60)
        print("\n📁 Output files:")
        print("   - features/font_features.csv (From STEP 2)")
        print("   - features/glyph_features.csv (Glyph features)")
        print("   - features/combined_features.csv (Merged features)")
        print("   - features/plots/glyph_analysis.png (Visualizations)")
        print("\nNEXT: Run STEP 4 - Implement Font Matching Detection")
    else:
        print("❌ Glyph feature extraction failed. Check PDF files.")