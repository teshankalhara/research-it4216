"""
STEP 3 SIMPLIFIED: Extract Glyph Features Directly from PDF Structure
No complex image processing - works with all PDFs
"""

import os
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
from collections import Counter
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "dataset"
FEATURES_PATH = "features/font_features.csv"
OUTPUT_PATH = "features/glyph_features.csv"

def extract_glyph_features_from_pdf(pdf_path):
    """
    Extract glyph/text features directly from PDF structure
    No image rendering - works reliably
    """
    try:
        doc = fitz.open(pdf_path)
        
        all_text = []
        char_counts = []
        line_lengths = []
        space_ratios = []
        word_lengths = []
        
        for page_num in range(min(len(doc), 3)):  # First 3 pages
            page = doc[page_num]
            text = page.get_text()
            
            if text.strip():
                all_text.append(text)
                
                # Character statistics
                char_count = len(text)
                char_counts.append(char_count)
                
                # Line analysis
                lines = text.split('\n')
                line_lengths.extend([len(line.strip()) for line in lines if line.strip()])
                
                # Space ratio (spacing patterns)
                space_count = text.count(' ')
                space_ratio = space_count / char_count if char_count > 0 else 0
                space_ratios.append(space_ratio)
                
                # Word length statistics
                words = text.split()
                word_lengths.extend([len(word) for word in words])
        
        # Get text block information from spans
        font_sizes = []
        char_widths = []
        
        for page_num in range(min(len(doc), 3)):
            page = doc[page_num]
            blocks = page.get_text("dict")
            
            for block in blocks.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # Get character level details
                        font_sizes.append(span.get("size", 0))
                        
                        # Get text length to estimate widths
                        text = span.get("text", "")
                        if text:
                            char_widths.append(len(text))
        
        doc.close()
        
        if not all_text:
            return None
        
        # Calculate features
        features = {
            # Text statistics
            'total_chars': np.mean(char_counts) if char_counts else 0,
            'char_count_std': np.std(char_counts) if len(char_counts) > 1 else 0,
            'avg_line_length': np.mean(line_lengths) if line_lengths else 0,
            'line_length_std': np.std(line_lengths) if len(line_lengths) > 1 else 0,
            'avg_word_length': np.mean(word_lengths) if word_lengths else 0,
            'word_length_std': np.std(word_lengths) if len(word_lengths) > 1 else 0,
            
            # Spacing analysis
            'avg_space_ratio': np.mean(space_ratios) if space_ratios else 0,
            'space_ratio_std': np.std(space_ratios) if len(space_ratios) > 1 else 0,
            
            # Font size analysis
            'avg_font_size': np.mean(font_sizes) if font_sizes else 0,
            'font_size_std': np.std(font_sizes) if len(font_sizes) > 1 else 0,
            
            # Character width variation
            'avg_char_width': np.mean(char_widths) if char_widths else 0,
            'char_width_std': np.std(char_widths) if len(char_widths) > 1 else 0,
            
            # Derived tampering indicators
            'text_density': np.mean(char_counts) / 1000 if char_counts else 0,
            'spacing_irregularity': np.std(space_ratios) if len(space_ratios) > 1 else 0,
            'font_size_variance': np.std(font_sizes) / (np.mean(font_sizes) + 0.01) if font_sizes else 0,
        }
        
        # Glyph consistency score (higher = more consistent)
        # Lower scores indicate possible tampering
        consistency_score = 1.0 - min(1.0, features['font_size_variance'] * 0.5 + features['spacing_irregularity'] * 2)
        
        features['glyph_consistency_score'] = max(0, min(1, consistency_score))
        
        return features
        
    except Exception as e:
        return None

def extract_all_glyph_features():
    """Extract glyph features from all PDFs"""
    
    print("="*60)
    print("STEP 3 SIMPLIFIED: EXTRACTING GLYPH/TEXT FEATURES")
    print("Analyzing text structure, spacing, and font patterns...")
    print("="*60)
    
    # Load font features
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
        if 'label' in df_glyph.columns and len(df_glyph) > 0:
            genuine = df_glyph[df_glyph['label'] == 'genuine']
            tampered = df_glyph[df_glyph['label'] == 'tampered']
            
            print(f"\n📊 Text/Glyph Features Comparison:")
            print(f"\n   Genuine PDFs (n={len(genuine)}):")
            print(f"     - Avg Line Length: {genuine['avg_line_length'].mean():.1f}")
            print(f"     - Avg Word Length: {genuine['avg_word_length'].mean():.2f}")
            print(f"     - Space Ratio: {genuine['avg_space_ratio'].mean():.4f}")
            print(f"     - Font Size Variance: {genuine['font_size_variance'].mean():.4f}")
            print(f"     - Glyph Consistency: {genuine['glyph_consistency_score'].mean():.3f}")
            
            print(f"\n   Tampered PDFs (n={len(tampered)}):")
            print(f"     - Avg Line Length: {tampered['avg_line_length'].mean():.1f}")
            print(f"     - Avg Word Length: {tampered['avg_word_length'].mean():.2f}")
            print(f"     - Space Ratio: {tampered['avg_space_ratio'].mean():.4f}")
            print(f"     - Font Size Variance: {tampered['font_size_variance'].mean():.4f}")
            print(f"     - Glyph Consistency: {tampered['glyph_consistency_score'].mean():.3f}")
            
            # Key differences
            print(f"\n   🔍 KEY DIFFERENCES (Tampered - Genuine):")
            print(f"     Font Size Variance: {tampered['font_size_variance'].mean() - genuine['font_size_variance'].mean():+.4f}")
            print(f"     Space Ratio: {tampered['avg_space_ratio'].mean() - genuine['avg_space_ratio'].mean():+.4f}")
            print(f"     Glyph Consistency: {tampered['glyph_consistency_score'].mean() - genuine['glyph_consistency_score'].mean():+.3f}")
        
        return df_glyph
    else:
        print("\n❌ No glyph features were extracted!")
        print("Creating backup features from font data...")
        
        # Create backup features from font data
        df_backup = df_font.copy()
        df_backup['glyph_consistency_score'] = 1 - df_backup['tampering_score_prelim']
        df_backup['avg_line_length'] = 0
        df_backup['avg_word_length'] = 0
        df_backup['avg_space_ratio'] = 0
        df_backup['font_size_variance'] = df_backup['font_variance']
        
        df_backup.to_csv(OUTPUT_PATH, index=False)
        return df_backup

def combine_and_detect():
    """Combine font + glyph features and perform tampering detection"""
    
    print("\n" + "="*60)
    print("STEP 4: FONT MATCHING & TAMPERING DETECTION")
    print("="*60)
    
    # Load features
    df_font = pd.read_csv("features/font_features.csv")
    
    glyph_path = "features/glyph_features.csv"
    if os.path.exists(glyph_path):
        df_glyph = pd.read_csv(glyph_path)
        print(f"✓ Loaded font features: {len(df_font)}")
        print(f"✓ Loaded glyph features: {len(df_glyph)}")
        
        # Merge features
        df_combined = pd.merge(df_font, df_glyph, on=['filename', 'label', 'tampering_type', 'doc_type'], how='left')
    else:
        print("⚠ Using font features only")
        df_combined = df_font.copy()
        df_combined['glyph_consistency_score'] = 1 - df_combined['tampering_score_prelim']
    
    # Fill NaN values
    df_combined = df_combined.fillna(0)
    
    # Calculate combined tampering score
    # Weight: 60% font consistency + 40% glyph consistency
    df_combined['detection_score'] = (
        df_combined['tampering_score_prelim'] * 0.6 +  # Font inconsistency
        (1 - df_combined['glyph_consistency_score']) * 0.4  # Glyph inconsistency
    )
    
    # Make prediction (threshold = 0.3)
    df_combined['prediction'] = (df_combined['detection_score'] > 0.3).astype(int)
    df_combined['predicted_label'] = df_combined['prediction'].map({0: 'genuine', 1: 'tampered'})
    
    # Calculate accuracy
    correct = (df_combined['predicted_label'] == df_combined['label']).sum()
    accuracy = correct / len(df_combined)
    
    # Calculate per-class metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
    
    y_true = (df_combined['label'] == 'tampered').astype(int)
    y_pred = df_combined['prediction']
    
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    
    print("\n" + "="*60)
    print("DETECTION RESULTS")
    print("="*60)
    print(f"\n📊 Overall Accuracy: {accuracy*100:.2f}%")
    print(f"📊 Precision: {precision*100:.2f}%")
    print(f"📊 Recall: {recall*100:.2f}%")
    print(f"📊 F1-Score: {f1*100:.2f}%")
    
    print(f"\n📈 Confusion Matrix:")
    print(f"   True Genuine: {cm[0,0]} correct, {cm[0,1]} false tampered")
    print(f"   True Tampered: {cm[1,1]} correct, {cm[1,0]} false genuine")
    
    # Performance by tampering type
    print(f"\n📊 Performance by Tampering Type:")
    for tamper_type in df_combined['tampering_type'].unique():
        subset = df_combined[df_combined['tampering_type'] == tamper_type]
        if len(subset) > 0:
            correct_subset = (subset['predicted_label'] == subset['label']).sum()
            acc_subset = correct_subset / len(subset)
            print(f"   {tamper_type}: {acc_subset*100:.1f}% ({correct_subset}/{len(subset)})")
    
    # Save results
    df_combined.to_csv("features/detection_results.csv", index=False)
    print(f"\n✓ Results saved: features/detection_results.csv")
    
    # Save metrics summary
    metrics = pd.DataFrame([{
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'true_negatives': cm[0,0],
        'false_positives': cm[0,1],
        'false_negatives': cm[1,0],
        'true_positives': cm[1,1]
    }])
    metrics.to_csv("features/metrics_summary.csv", index=False)
    
    return df_combined, metrics

def visualize_detection_results(df_combined):
    """Create visualizations of detection results"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs("features/plots", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Detection score distribution
        ax1 = axes[0, 0]
        for label, color in [('genuine', 'green'), ('tampered', 'red')]:
            data = df_combined[df_combined['label'] == label]['detection_score']
            if len(data) > 0:
                ax1.hist(data, alpha=0.5, bins=30, label=label, color=color)
        ax1.axvline(x=0.3, color='black', linestyle='--', label='Threshold (0.3)')
        ax1.set_xlabel('Detection Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Detection Score Distribution')
        ax1.legend()
        
        # Plot 2: Accuracy by tampering type
        ax2 = axes[0, 1]
        tamper_results = df_combined[df_combined['tampering_type'] != 'none'].groupby('tampering_type').apply(
            lambda x: (x['predicted_label'] == x['label']).mean()
        )
        tamper_results.plot(kind='bar', ax=ax2, color='skyblue')
        ax2.set_xlabel('Tampering Type')
        ax2.set_ylabel('Detection Accuracy')
        ax2.set_title('Accuracy by Tampering Method')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        # Plot 3: Font vs Glyph contribution
        ax3 = axes[1, 0]
        sample = df_combined.sample(min(500, len(df_combined)))
        ax3.scatter(sample['tampering_score_prelim'], 1 - sample['glyph_consistency_score'], 
                   c=sample['detection_score'], cmap='RdYlGn', alpha=0.5)
        ax3.set_xlabel('Font Inconsistency Score')
        ax3.set_ylabel('Glyph Inconsistency Score')
        ax3.set_title('Font vs Glyph Inconsistency (Color = Detection Score)')
        
        # Plot 4: Confusion Matrix Heatmap
        ax4 = axes[1, 1]
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix((df_combined['label'] == 'tampered').astype(int), 
                             df_combined['prediction'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                    xticklabels=['Predicted Genuine', 'Predicted Tampered'],
                    yticklabels=['Actual Genuine', 'Actual Tampered'])
        ax4.set_title('Confusion Matrix')
        
        plt.tight_layout()
        plt.savefig("features/plots/detection_results.png", dpi=150)
        print(f"✓ Visualization saved: features/plots/detection_results.png")
        plt.close()
        
    except Exception as e:
        print(f"Visualization error: {e}")

if __name__ == "__main__":
    # Extract glyph features
    df_glyph = extract_all_glyph_features()
    
    # Perform detection
    df_results, metrics = combine_and_detect()
    
    # Visualize
    visualize_detection_results(df_results)
    
    print("\n" + "="*60)
    print("✅ RESEARCH PIPELINE COMPLETE!")
    print("="*60)
    print("\n📁 Final outputs:")
    print("   - features/font_features.csv (Font metadata)")
    print("   - features/glyph_features.csv (Text features)")
    print("   - features/detection_results.csv (Predictions)")
    print("   - features/metrics_summary.csv (Performance metrics)")
    print("   - features/plots/detection_results.png (Visualizations)")
    
    print("\n📊 Final Performance Summary:")
    print(f"   Accuracy: {metrics['accuracy'].iloc[0]*100:.2f}%")
    print(f"   Precision: {metrics['precision'].iloc[0]*100:.2f}%")
    print(f"   Recall: {metrics['recall'].iloc[0]*100:.2f}%")
    print(f"   F1-Score: {metrics['f1_score'].iloc[0]*100:.2f}%")