"""
PDF Tampering Detection Engine
Uses your trained Random Forest model
"""

import os
import pickle
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
from collections import Counter
import tempfile

class PDFTamperingDetector:
    """Main detection class for PDF tampering using trained model"""
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.load_model()
    
    def load_model(self):
        """Load pre-trained Random Forest model and scaler"""
        try:
            model_path = os.path.join(os.path.dirname(__file__), "model/random_forest_model.pkl")
            scaler_path = os.path.join(os.path.dirname(__file__), "model/scaler.pkl")
            features_path = os.path.join(os.path.dirname(__file__), "model/feature_columns.pkl")
            
            if os.path.exists(model_path):
                with open(model_path, "rb") as f:
                    self.model = pickle.load(f)
                print("✓ Loaded trained Random Forest model")
            else:
                print("⚠ Model file not found. Run train_model.py first.")
                self.model = None
            
            if os.path.exists(scaler_path):
                with open(scaler_path, "rb") as f:
                    self.scaler = pickle.load(f)
                print("✓ Loaded scaler")
            
            if os.path.exists(features_path):
                with open(features_path, "rb") as f:
                    self.feature_cols = pickle.load(f)
                print(f"✓ Loaded {len(self.feature_cols)} feature columns")
            
            self.model_loaded = self.model is not None
            
        except Exception as e:
            print(f"Model load error: {e}")
            self.model_loaded = False
            self.model = None
    
    def extract_font_features_from_pdf(self, pdf_path):
        """Extract font metadata features from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            font_names = []
            font_sizes = []
            
            for page_num in range(min(len(doc), 3)):
                page = doc[page_num]
                blocks = page.get_text("dict")
                
                for block in blocks.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_name = span.get("font", "unknown")
                            font_size = span.get("size", 0)
                            text = span.get("text", "").strip()
                            
                            if text:
                                font_names.append(font_name)
                                font_sizes.append(font_size)
            
            doc.close()
            
            if font_names:
                font_counter = Counter(font_names)
                most_common_count = font_counter.most_common(1)[0][1]
                dominant_font_ratio = most_common_count / len(font_names)
                unique_fonts = len(set(font_names))
                font_variance = unique_fonts / len(font_names) if font_names else 0
                tampering_score = 1 - dominant_font_ratio
            else:
                dominant_font_ratio = 0.99
                unique_fonts = 1
                font_variance = 0
                tampering_score = 0.01
            
            return {
                'dominant_font_ratio': dominant_font_ratio,
                'unique_fonts': unique_fonts,
                'font_variance': font_variance,
                'tampering_score_prelim': tampering_score
            }
        except Exception as e:
            print(f"Font extraction error: {e}")
            return None
    
    def extract_text_features_from_pdf(self, pdf_path):
        """Extract text structure features from PDF"""
        try:
            doc = fitz.open(pdf_path)
            
            all_text = ""
            lines_list = []
            words_list = []
            space_count = 0
            total_chars = 0
            font_sizes = []
            
            for page_num in range(min(len(doc), 3)):
                page = doc[page_num]
                text = page.get_text()
                
                if text:
                    all_text += text
                    
                    # Line analysis
                    lines = text.split('\n')
                    lines = [l.strip() for l in lines if l.strip()]
                    lines_list.extend([len(l) for l in lines])
                    
                    # Word analysis
                    words = text.split()
                    words_list.extend([len(w) for w in words])
                    
                    # Space ratio
                    space_count += text.count(' ')
                    total_chars += len(text)
                
                # Get font sizes from spans
                blocks = page.get_text("dict")
                for block in blocks.get("blocks", []):
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            font_sizes.append(span.get("size", 0))
            
            doc.close()
            
            if all_text.strip():
                avg_line_length = np.mean(lines_list) if lines_list else 30
                avg_word_length = np.mean(words_list) if words_list else 7
                space_ratio = space_count / total_chars if total_chars > 0 else 0.15
                font_size_variance = np.var(font_sizes) if len(font_sizes) > 1 else 0
                space_ratio_std = 0  # Simplified for now
            else:
                avg_line_length = 30
                avg_word_length = 7
                space_ratio = 0.15
                font_size_variance = 0
                space_ratio_std = 0
            
            return {
                'avg_line_length': avg_line_length,
                'avg_word_length': avg_word_length,
                'avg_space_ratio': space_ratio,
                'font_size_variance': font_size_variance,
                'space_ratio_std': space_ratio_std
            }
        except Exception as e:
            print(f"Text extraction error: {e}")
            return None
    
    def extract_all_features(self, pdf_path):
        """Extract all features needed for model prediction"""
        
        font_features = self.extract_font_features_from_pdf(pdf_path)
        text_features = self.extract_text_features_from_pdf(pdf_path)
        
        if font_features is None or text_features is None:
            return None
        
        # Combine all features
        all_features = {**font_features, **text_features}
        
        # Create dataframe with proper feature order
        if self.feature_cols:
            feature_dict = {}
            for col in self.feature_cols:
                feature_dict[col] = all_features.get(col, 0)
            return feature_dict
        else:
            return all_features
    
    def detect(self, pdf_path):
        """Main detection function using trained model"""
        
        # Extract features
        features_dict = self.extract_all_features(pdf_path)
        
        if features_dict is None:
            return {
                'is_tampered': False,
                'confidence': 0,
                'error': 'Could not process PDF',
                'details': {},
                'issues': [],
                'recommendation': 'Error processing PDF. Please try another file.'
            }
        
        # If model is loaded, use it for prediction
        if self.model_loaded and self.feature_cols:
            try:
                # Create feature vector in correct order
                feature_vector = []
                for col in self.feature_cols:
                    feature_vector.append(features_dict.get(col, 0))
                
                # Convert to numpy array and scale
                X = np.array(feature_vector).reshape(1, -1)
                
                if self.scaler:
                    X_scaled = self.scaler.transform(X)
                else:
                    X_scaled = X
                
                # Predict
                prediction = self.model.predict(X_scaled)[0]
                probabilities = self.model.predict_proba(X_scaled)[0]
                
                is_tampered = bool(prediction == 1)
                confidence = max(probabilities) * 100
                
            except Exception as e:
                print(f"Model prediction error: {e}")
                # Fallback to rule-based
                return self.rule_based_detect(features_dict)
        else:
            # Fallback to rule-based detection
            return self.rule_based_detect(features_dict)
        
        # Prepare details for display
        details = {
            'font_consistency': f"{features_dict.get('dominant_font_ratio', 0)*100:.1f}%",
            'unique_fonts': features_dict.get('unique_fonts', 0),
            'font_variance': f"{features_dict.get('font_variance', 0):.3f}",
            'avg_line_length': f"{features_dict.get('avg_line_length', 0):.1f} chars",
            'avg_word_length': f"{features_dict.get('avg_word_length', 0):.1f} chars",
            'space_ratio': f"{features_dict.get('avg_space_ratio', 0)*100:.1f}%",
            'model_confidence': f"{confidence:.1f}%"
        }
        
        # Find specific issues
        issues = []
        if features_dict.get('dominant_font_ratio', 1) < 0.95:
            issues.append("Multiple fonts detected - possible font tampering")
        if features_dict.get('unique_fonts', 0) > 2:
            issues.append("Unusual font variation - possible embedding mismatch")
        if features_dict.get('avg_space_ratio', 0) > 0.16:
            issues.append("Abnormal spacing pattern - possible character replacement")
        if features_dict.get('avg_word_length', 7) < 6 or features_dict.get('avg_word_length', 7) > 8:
            issues.append("Irregular word length distribution")
        
        return {
            'is_tampered': is_tampered,
            'confidence': confidence,
            'prediction': 'Tampered' if is_tampered else 'Genuine',
            'details': details,
            'issues': issues,
            'recommendation': '⚠️ This document appears TAMPERED. Please verify the highlighted sections.' if is_tampered else '✅ This document appears GENUINE. No tampering detected.',
            'probabilities': {
                'genuine': probabilities[0],
                'tampered': probabilities[1]
            }
        }
    
    def rule_based_detect(self, features_dict):
        """Fallback rule-based detection when model not available"""
        
        score = 0
        
        if features_dict.get('dominant_font_ratio', 1) < 0.95:
            score += 40
        if features_dict.get('unique_fonts', 0) > 1:
            score += 20
        if features_dict.get('font_variance', 0) > 0.05:
            score += 10
        if features_dict.get('avg_space_ratio', 0) > 0.16:
            score += 15
        if features_dict.get('avg_word_length', 7) < 6.5 or features_dict.get('avg_word_length', 7) > 7.5:
            score += 15
        
        is_tampered = score >= 50
        confidence = min(95, max(50, score)) if is_tampered else min(95, max(50, 100 - score))
        
        details = {
            'font_consistency': f"{features_dict.get('dominant_font_ratio', 0)*100:.1f}%",
            'unique_fonts': features_dict.get('unique_fonts', 0),
            'font_variance': f"{features_dict.get('font_variance', 0):.3f}",
            'avg_line_length': f"{features_dict.get('avg_line_length', 0):.1f} chars",
            'avg_word_length': f"{features_dict.get('avg_word_length', 0):.1f} chars",
            'space_ratio': f"{features_dict.get('avg_space_ratio', 0)*100:.1f}%",
            'tampering_score': f"{score:.0f}/100"
        }
        
        issues = []
        if features_dict.get('dominant_font_ratio', 1) < 0.95:
            issues.append("Multiple fonts detected - possible font tampering")
        if features_dict.get('avg_space_ratio', 0) > 0.16:
            issues.append("Abnormal spacing pattern - possible character replacement")
        
        return {
            'is_tampered': is_tampered,
            'confidence': confidence,
            'details': details,
            'issues': issues,
            'recommendation': '⚠️ This document appears TAMPERED.' if is_tampered else '✅ This document appears GENUINE.'
        }

# Singleton instance
detector = PDFTamperingDetector()

def analyze_pdf(pdf_path):
    """Public API for PDF analysis"""
    return detector.detect(pdf_path)