"""
STEP 2: Extract Font Metadata from PDF Dataset
Extracts: font name, size, encoding, type, and consistency metrics
"""

import os
import pandas as pd
import fitz  # PyMuPDF
from collections import Counter
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATASET_PATH = "dataset"
GROUND_TRUTH_PATH = "dataset/ground_truth/labels.csv"
OUTPUT_PATH = "features/font_features.csv"

def extract_font_metadata(pdf_path):
    """
    Extract comprehensive font metadata from a PDF file
    
    Returns:
        dict: Font features including names, sizes, consistency metrics
    """
    try:
        doc = fitz.open(pdf_path)
        
        font_data = {
            'font_names': [],
            'font_sizes': [],
            'font_encodings': [],
            'font_types': [],  # Type1, TrueType, etc.
            'per_page_fonts': [],
            'total_pages': len(doc),
            'text_blocks': 0
        }
        
        for page_num, page in enumerate(doc):
            # Get text blocks with font info
            blocks = page.get_text("dict")
            
            page_fonts = []
            
            for block in blocks.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # Extract font details
                        font_name = span.get("font", "unknown")
                        font_size = span.get("size", 0)
                        font_flags = span.get("flags", 0)
                        text = span.get("text", "").strip()
                        
                        if text:  # Only record if there's actual text
                            font_data['font_names'].append(font_name)
                            font_data['font_sizes'].append(font_size)
                            font_data['text_blocks'] += 1
                            page_fonts.append(font_name)
                            
                            # Determine font type based on flags
                            if font_flags & 1 << 0:  # Fixed pitch
                                font_type = "Monospaced"
                            elif font_flags & 1 << 3:  # Serif
                                font_type = "Serif"
                            else:
                                font_type = "Sans-Serif"
                            font_data['font_types'].append(font_type)
            
            font_data['per_page_fonts'].append(page_fonts)
        
        doc.close()
        
        # Calculate derived features
        if font_data['font_names']:
            # Most common font (dominant font)
            font_counter = Counter(font_data['font_names'])
            most_common_font = font_counter.most_common(1)[0][0] if font_counter else "none"
            most_common_font_count = font_counter.most_common(1)[0][1] if font_counter else 0
            
            # Font diversity (number of unique fonts)
            unique_fonts = len(set(font_data['font_names']))
            
            # Font consistency score (percentage of document using dominant font)
            font_consistency = most_common_font_count / len(font_data['font_names']) if font_data['font_names'] else 0
            
            # Font size statistics
            size_counter = Counter(font_data['font_sizes'])
            most_common_size = size_counter.most_common(1)[0][0] if size_counter else 0
            unique_sizes = len(set(font_data['font_sizes']))
            avg_font_size = sum(font_data['font_sizes']) / len(font_data['font_sizes']) if font_data['font_sizes'] else 0
            
            # Check for font inconsistencies (suspicious patterns)
            # Multiple different fonts in same page or across pages
            font_variance = unique_fonts / len(font_data['font_names']) if font_data['font_names'] else 0
            
            # Detect potential tampering indicators
            tampering_indicators = {
                'multiple_fonts_same_page': 0,
                'font_size_anomalies': 0,
                'rare_fonts': []
            }
            
            # Check each page for multiple fonts
            for page_fonts in font_data['per_page_fonts']:
                if len(set(page_fonts)) > 2:  # More than 2 fonts on same page
                    tampering_indicators['multiple_fonts_same_page'] += 1
            
            # Find rare fonts (used less than 1% of text)
            total_spans = len(font_data['font_names'])
            for font, count in font_counter.items():
                if count / total_spans < 0.01 and font != most_common_font:
                    tampering_indicators['rare_fonts'].append(font)
            
        else:
            # No text found
            most_common_font = "none"
            unique_fonts = 0
            font_consistency = 0
            most_common_size = 0
            unique_sizes = 0
            avg_font_size = 0
            font_variance = 0
            tampering_indicators = {}
        
        return {
            'total_pages': font_data['total_pages'],
            'text_blocks': font_data['text_blocks'],
            'unique_fonts': unique_fonts,
            'dominant_font': most_common_font,
            'dominant_font_ratio': font_consistency,
            'font_variance': font_variance,
            'unique_font_sizes': unique_sizes,
            'dominant_font_size': most_common_size,
            'avg_font_size': avg_font_size,
            'multiple_fonts_per_page': tampering_indicators.get('multiple_fonts_same_page', 0),
            'rare_fonts_count': len(tampering_indicators.get('rare_fonts', [])),
            'tampering_score_prelim': 1 - font_consistency if font_consistency > 0 else 0  # Higher = more suspicious
        }
        
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return None

def process_all_pdfs():
    """Extract font features from all PDFs in dataset"""
    
    print("="*60)
    print("STEP 2: EXTRACTING FONT METADATA")
    print("Processing 2,500 PDFs...")
    print("="*60)
    
    # Load ground truth
    if not os.path.exists(GROUND_TRUTH_PATH):
        print(f"Error: Ground truth not found at {GROUND_TRUTH_PATH}")
        return None
    
    df_truth = pd.read_csv(GROUND_TRUTH_PATH)
    print(f"✓ Loaded ground truth: {len(df_truth)} PDFs")
    
    # Create features directory
    os.makedirs("features", exist_ok=True)
    
    # Process each PDF
    all_features = []
    
    for idx, row in tqdm(df_truth.iterrows(), total=len(df_truth), desc="Extracting font features"):
        pdf_path = os.path.join(DATASET_PATH, row['filename'])
        
        if os.path.exists(pdf_path):
            features = extract_font_metadata(pdf_path)
            
            if features:
                features['filename'] = row['filename']
                features['label'] = row['label']
                features['tampering_type'] = row['tampering_type']
                features['doc_type'] = row['doc_type']
                all_features.append(features)
        else:
            print(f"Warning: File not found - {pdf_path}")
    
    # Convert to DataFrame
    df_features = pd.DataFrame(all_features)
    
    # Save features
    df_features.to_csv(OUTPUT_PATH, index=False)
    
    print("\n" + "="*60)
    print("FONT FEATURE EXTRACTION SUMMARY")
    print("="*60)
    print(f"✓ Processed: {len(df_features)} PDFs")
    print(f"✓ Features saved: {OUTPUT_PATH}")
    
    # Statistics by label
    print("\n📊 Font Features by Document Type:")
    genuine = df_features[df_features['label'] == 'genuine']
    tampered = df_features[df_features['label'] == 'tampered']
    
    print(f"\n   Genuine PDFs (n={len(genuine)}):")
    print(f"     - Avg unique fonts: {genuine['unique_fonts'].mean():.2f}")
    print(f"     - Avg font consistency: {genuine['dominant_font_ratio'].mean():.3f}")
    print(f"     - Avg font variance: {genuine['font_variance'].mean():.3f}")
    
    print(f"\n   Tampered PDFs (n={len(tampered)}):")
    print(f"     - Avg unique fonts: {tampered['unique_fonts'].mean():.2f}")
    print(f"     - Avg font consistency: {tampered['dominant_font_ratio'].mean():.3f}")
    print(f"     - Avg font variance: {tampered['font_variance'].mean():.3f}")
    
    print(f"\n   Key Difference:")
    print(f"     Font consistency: Genuine {genuine['dominant_font_ratio'].mean():.3f} vs Tampered {tampered['dominant_font_ratio'].mean():.3f}")
    print(f"     Font variance: Genuine {genuine['font_variance'].mean():.3f} vs Tampered {tampered['font_variance'].mean():.3f}")
    
    return df_features

def visualize_font_features(df_features):
    """Create visualizations of font features"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        os.makedirs("features/plots", exist_ok=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Plot 1: Font consistency distribution
        ax1 = axes[0, 0]
        for label, color in [('genuine', 'green'), ('tampered', 'red')]:
            data = df_features[df_features['label'] == label]['dominant_font_ratio']
            ax1.hist(data, alpha=0.5, bins=30, label=label, color=color)
        ax1.set_xlabel('Font Consistency Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Font Consistency: Genuine vs Tampered')
        ax1.legend()
        
        # Plot 2: Unique fonts distribution
        ax2 = axes[0, 1]
        for label, color in [('genuine', 'green'), ('tampered', 'red')]:
            data = df_features[df_features['label'] == label]['unique_fonts']
            ax2.hist(data, alpha=0.5, bins=30, label=label, color=color)
        ax2.set_xlabel('Number of Unique Fonts')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Font Diversity: Genuine vs Tampered')
        ax2.legend()
        
        # Plot 3: Box plot comparison
        ax3 = axes[1, 0]
        sns.boxplot(x='label', y='tampering_score_prelim', data=df_features, ax=ax3)
        ax3.set_xlabel('Document Type')
        ax3.set_ylabel('Tampering Score (Preliminary)')
        ax3.set_title('Preliminary Tampering Score by Document Type')
        
        # Plot 4: Feature correlation
        ax4 = axes[1, 1]
        numeric_cols = ['unique_fonts', 'dominant_font_ratio', 'font_variance', 'unique_font_sizes', 'tampering_score_prelim']
        corr = df_features[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax4)
        ax4.set_title('Feature Correlations')
        
        plt.tight_layout()
        plt.savefig("features/plots/font_analysis.png", dpi=150)
        print(f"✓ Visualization saved: features/plots/font_analysis.png")
        plt.close()
        
    except Exception as e:
        print(f"Visualization skipped (install matplotlib/seaborn if needed): {e}")

if __name__ == "__main__":
    # Install required library if not present
    try:
        import fitz
    except ImportError:
        print("Installing PyMuPDF...")
        os.system("pip install PyMuPDF")
        import fitz
    
    # Extract features
    df_features = process_all_pdfs()
    
    if df_features is not None:
        # Create visualizations
        try:
            import matplotlib
            import seaborn
            visualize_font_features(df_features)
        except ImportError:
            print("\n⚠ For visualizations, install: pip install matplotlib seaborn")
        
        # Show sample features
        print("\n" + "="*60)
        print("SAMPLE FONT FEATURES (First 5 PDFs)")
        print("="*60)
        display_cols = ['filename', 'label', 'unique_fonts', 'dominant_font_ratio', 'tampering_score_prelim']
        print(df_features[display_cols].head(10))
        
        print("\n" + "="*60)
        print("✅ STEP 2 COMPLETE!")
        print("="*60)
        print("\n📁 Output files:")
        print("   - features/font_features.csv (All font metadata)")
        print("   - features/plots/font_analysis.png (Visualizations)")
        print("\nNEXT: Run STEP 3 - Extract Visual Glyph Features")
    else:
        print("❌ Feature extraction failed")