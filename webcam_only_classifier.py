# ==============================================================================
# WEBCAM ONLY FRUIT CLASSIFIER - COMPATIBLE WITH IMPROVED MODEL
# Load pre-trained model dan jalankan real-time classification
# ==============================================================================

import cv2
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# REQUIRED CLASSES FROM IMPROVED MODEL
# =============================================================================

class FruitFeatureExtractor:
    """
    Feature Extractor untuk ekstraksi fitur handcrafted dari citra buah
    (Copy dari improved model untuk compatibility)
    """
    
    def __init__(self):
        self.feature_names = [
            "Mean H", "Mean S", "Mean V",           # HSV means
            "Std H", "Std S", "Std V",              # HSV standard deviations
            "Ratio S/H", "Ratio V/S",               # Color ratios
            "Entropy H", "Prop Kuning", "Prop Hijau", # Distribution features
            "Std Gray"                              # Texture feature
        ]
    
    def extract_features(self, image_path_or_array):
        """
        Ekstraksi 12 fitur handcrafted dari citra buah
        """
        # Load image
        if isinstance(image_path_or_array, str):
            if not os.path.exists(image_path_or_array):
                raise FileNotFoundError(f"Image not found: {image_path_or_array}")
            image = cv2.imread(image_path_or_array)
        else:
            image = image_path_or_array.copy()
            
        if image is None:
            raise ValueError("Cannot read image")
        
        # Convert to HSV
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # 1. HSV STATISTICAL FEATURES (6 fitur)
        mean_h = np.mean(h)
        mean_s = np.mean(s)  
        mean_v = np.mean(v)
        std_h = np.std(h)
        std_s = np.std(s)
        std_v = np.std(v)
        
        # 2. COLOR RATIO FEATURES (2 fitur)
        ratio_s_h = mean_s / (mean_h + 1e-5)
        ratio_v_s = mean_v / (mean_s + 1e-5)
        
        # 3. HUE ENTROPY FEATURE (1 fitur)
        hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
        hist_h = hist_h / (hist_h.sum() + 1e-10)
        entropy_h = -np.sum(hist_h * np.log2(hist_h + 1e-10))
        
        # 4. COLOR PROPORTION FEATURES (2 fitur)
        total_pixels = h.size
        prop_kuning = np.sum((h >= 20) & (h <= 40)) / total_pixels
        prop_hijau = np.sum((h >= 50) & (h <= 70)) / total_pixels
        
        # 5. TEXTURE FEATURE (1 fitur)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        std_gray = np.std(gray)
        
        features = np.array([
            mean_h, mean_s, mean_v,
            std_h, std_s, std_v,
            ratio_s_h, ratio_v_s,
            entropy_h, prop_kuning, prop_hijau,
            std_gray
        ])
        
        return features

# =============================================================================
# WEBCAM CLASSIFIER
# =============================================================================

class WebcamFruitClassifier:
    """
    Webcam classifier untuk load model yang sudah ditraining dari improved system
    """
    
    def __init__(self, model_path='fruit_ripeness_model_improved.pkl'):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.feature_extractor = None
        self.selected_features = None
        self.all_features = None
        self.model_name = None
        self.model_results = None
        self.prediction_history = []
        
        # Debug control
        self.debug_count = 0
        self.max_debug_messages = 5  # Show debug messages only for first 5 predictions
        
        self.load_model()
    
    def load_model(self):
        """Load trained model dan komponen preprocessing dari improved system"""
        try:
            print(f"📦 Loading improved model from {self.model_path}...")
            
            # Try loading with custom objects
            try:
                model_data = joblib.load(self.model_path)
            except AttributeError as e:
                print(f"⚠️  Compatibility issue detected: {e}")
                print("🔧 Attempting to fix compatibility...")
                
                # Create global reference for joblib
                import sys
                sys.modules['__main__'].FruitFeatureExtractor = FruitFeatureExtractor
                
                # Try loading again
                model_data = joblib.load(self.model_path)
            
            # Load all components from improved model
            self.model = model_data.get('model')
            self.scaler = model_data.get('scaler')
            self.feature_selector = model_data.get('feature_selector')
            self.label_encoder = model_data.get('label_encoder')
            
            # Handle feature extractor
            if 'feature_extractor' in model_data:
                self.feature_extractor = model_data['feature_extractor']
            else:
                print("⚠️  Feature extractor not found in model, creating new one...")
                self.feature_extractor = FruitFeatureExtractor()
            
            self.selected_features = model_data.get('selected_features', [])
            self.all_features = model_data.get('all_features', [])
            self.model_name = model_data.get('model_name', 'Unknown')
            self.model_results = model_data.get('model_results', {})
            
            # Validate essential components
            if self.model is None:
                raise ValueError("Model not found in saved file")
            if self.label_encoder is None:
                raise ValueError("Label encoder not found in saved file")
            if not self.selected_features:
                print("⚠️  Selected features not found, using all features...")
                self.selected_features = self.feature_extractor.feature_names.copy()
            
            print(f"✅ Improved model loaded successfully!")
            print(f"   Model Type: {self.model_name}")
            print(f"   Test Accuracy: {self.model_results.get('test_accuracy', 0.0):.1%}")
            print(f"   F1 Score: {self.model_results.get('f1_score', 0.0):.1%}")
            print(f"   AUC Score: {self.model_results.get('auc_score', 0.0):.1%}")
            print(f"   Total Features: {len(self.all_features) if self.all_features else 'Unknown'}")
            print(f"   Selected Features: {len(self.selected_features)}")
            print(f"   Features used: {self.selected_features}")
            
        except FileNotFoundError:
            print(f"❌ Model file {self.model_path} not found!")
            print("Please run the improved training script first:")
            print("  python fruit_classification_improved.py")
            raise
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("🔧 Troubleshooting suggestions:")
            print("1. Make sure the model was saved with the same class definitions")
            print("2. Try re-training the model if compatibility issues persist")
            print("3. Check if all required libraries are installed")
            raise
    
    def extract_and_select_features(self, roi):
        """Extract features dan apply feature selection sesuai training"""
        try:
            # Extract all features using feature extractor
            all_features = self.feature_extractor.extract_features(roi)
            
            # Debug only for first few predictions
            debug_mode = self.debug_count < self.max_debug_messages
            
            if debug_mode:
                print(f"Debug {self.debug_count+1}: Extracted {len(all_features)} features")
                print(f"Debug {self.debug_count+1}: Available features: {self.feature_extractor.feature_names}")
                print(f"Debug {self.debug_count+1}: Selected features: {self.selected_features}")
            
            # Check if we have the right number of features
            if len(all_features) != len(self.feature_extractor.feature_names):
                if debug_mode:
                    print(f"Warning: Feature count mismatch! Expected {len(self.feature_extractor.feature_names)}, got {len(all_features)}")
                # Pad or truncate if necessary
                if len(all_features) < len(self.feature_extractor.feature_names):
                    all_features = np.pad(all_features, (0, len(self.feature_extractor.feature_names) - len(all_features)))
                else:
                    all_features = all_features[:len(self.feature_extractor.feature_names)]
            
            # Create feature dictionary for easy mapping
            feature_dict = dict(zip(self.feature_extractor.feature_names, all_features))
            
            # Handle selected features
            if self.selected_features:
                # Select only the features that were used during training
                selected_feature_values = []
                for name in self.selected_features:
                    if name in feature_dict:
                        selected_feature_values.append(feature_dict[name])
                    else:
                        if debug_mode:
                            print(f"Warning: Selected feature '{name}' not found in extracted features")
                        selected_feature_values.append(0.0)  # Default value
                
                selected_feature_values = np.array(selected_feature_values)
            else:
                # If no selected features specified, use all features
                if debug_mode:
                    print("Warning: No selected features specified, using all features")
                selected_feature_values = all_features
            
            if debug_mode:
                print(f"Debug {self.debug_count+1}: Final selected feature values shape: {selected_feature_values.shape}")
            
            return selected_feature_values, all_features
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            # Return zeros with correct dimensions
            n_selected = len(self.selected_features) if self.selected_features else 12
            n_all = len(self.feature_extractor.feature_names)
            return np.zeros(n_selected), np.zeros(n_all)
    
    def predict_ripeness(self, roi):
        """Predict fruit ripeness dari ROI menggunakan improved model"""
        try:
            self.debug_count += 1
            debug_mode = self.debug_count <= self.max_debug_messages
            
            # Extract and select features
            selected_features, all_features = self.extract_and_select_features(roi)
            
            # Handle scaling carefully
            if self.scaler is not None:
                # Check if scaler expects different number of features
                expected_features = self.scaler.n_features_in_ if hasattr(self.scaler, 'n_features_in_') else len(selected_features)
                
                if debug_mode:
                    print(f"Debug {self.debug_count}: Scaler expects {expected_features} features, we have {len(selected_features)}")
                
                if len(selected_features) != expected_features:
                    if debug_mode:
                        print(f"Info: Adjusting feature dimensions from {len(selected_features)} to {expected_features}")
                    
                    # Try to fix the mismatch
                    if expected_features == 12 and len(all_features) == 12:
                        # Use all features if scaler was trained on all features
                        features_for_scaling = all_features
                        if debug_mode:
                            print("Using all 12 features for scaling")
                    elif len(selected_features) > expected_features:
                        # Truncate
                        features_for_scaling = selected_features[:expected_features]
                        if debug_mode:
                            print(f"Truncating features to {expected_features}")
                    else:
                        # Pad with median values from all_features
                        if len(all_features) >= expected_features:
                            features_for_scaling = all_features[:expected_features]
                        else:
                            # Pad with zeros as last resort
                            features_for_scaling = np.pad(selected_features, (0, expected_features - len(selected_features)))
                        if debug_mode:
                            print(f"Padding/adjusting features to {expected_features}")
                else:
                    features_for_scaling = selected_features
                
                # Scale features
                features_scaled = self.scaler.transform(features_for_scaling.reshape(1, -1))
                if debug_mode:
                    print(f"Debug {self.debug_count}: Scaled features shape: {features_scaled.shape}")
            else:
                features_scaled = selected_features.reshape(1, -1)
                if debug_mode:
                    print("Debug: No scaler found, using raw features")
            
            # Apply feature selection if it was used during training
            if self.feature_selector is not None:
                if debug_mode:
                    print(f"Debug {self.debug_count}: Applying feature selector...")
                try:
                    features_final = self.feature_selector.transform(features_scaled)
                    if debug_mode:
                        print(f"Debug {self.debug_count}: After feature selection: {features_final.shape}")
                except Exception as selector_error:
                    if debug_mode:
                        print(f"Warning: Feature selector failed: {selector_error}")
                    features_final = features_scaled
            else:
                features_final = features_scaled
                if debug_mode:
                    print("Debug: No feature selector found")
            
            # Show completion message after debug period
            if self.debug_count == self.max_debug_messages:
                print(f"✅ Pipeline configured successfully! Debug messages disabled for better performance.")
            
            # Make prediction based on model type
            if hasattr(self.model, 'predict_proba'):
                # Traditional sklearn models
                prediction_encoded = self.model.predict(features_final)[0]
                probabilities = self.model.predict_proba(features_final)[0]
                confidence = np.max(probabilities)
                
                # Decode prediction using label encoder
                prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
                
            elif hasattr(self.model, 'predict') and 'tensorflow' in str(type(self.model)):
                # TensorFlow/Keras model
                prediction_proba = self.model.predict(features_final, verbose=0)[0]
                
                # Handle different output shapes
                if hasattr(prediction_proba, '__len__') and len(prediction_proba) > 1:
                    prediction_encoded = np.argmax(prediction_proba)
                    confidence = np.max(prediction_proba)
                else:
                    # Binary classification with sigmoid
                    prediction_encoded = int(prediction_proba > 0.5)
                    confidence = prediction_proba if prediction_proba > 0.5 else (1 - prediction_proba)
                
                # Decode prediction
                prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
                
            else:
                # Fallback for other model types
                prediction_encoded = self.model.predict(features_final)[0]
                prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
                confidence = 0.8  # Default confidence
            
            return prediction, confidence, selected_features, all_features
            
        except Exception as e:
            # Only print error details in debug mode to avoid spam
            if self.debug_count <= self.max_debug_messages:
                print(f"Prediction error: {e}")
                import traceback
                traceback.print_exc()
            
            return "Error", 0.0, np.zeros(len(self.selected_features) if self.selected_features else 12), np.zeros(12)
    
    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions using history untuk stabilitas"""
        self.prediction_history.append((prediction, confidence))
        
        # Keep only last 10 predictions
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
        
        # Get most common prediction in recent history
        recent_predictions = [p[0] for p in self.prediction_history[-5:]]
        recent_confidences = [p[1] for p in self.prediction_history[-5:]]
        
        # Most frequent prediction
        if recent_predictions:
            smooth_prediction = max(set(recent_predictions), key=recent_predictions.count)
            avg_confidence = np.mean(recent_confidences)
        else:
            smooth_prediction = prediction
            avg_confidence = confidence
        
        return smooth_prediction, avg_confidence
    
    def run_classification(self):
        """Main webcam classification loop"""
        print("\n" + "🎥" * 30)
        print("REAL-TIME FRUIT RIPENESS CLASSIFICATION")
        print("🎥" * 30)
        print(f"Model: {self.model_name}")
        print(f"Test Accuracy: {self.model_results.get('test_accuracy', 0.0):.1%}")
        print(f"F1 Score: {self.model_results.get('f1_score', 0.0):.1%}")
        print(f"Selected Features: {len(self.selected_features)}")
        print("\nControls:")
        print("  'q' or ESC : Quit application")
        print("  's'        : Save current prediction")
        print("  'r'        : Reset prediction history")
        print("  'h'        : Toggle feature display")
        print("  'f'        : Toggle fullscreen mode")
        print("  'a'        : Toggle analysis mode")
        print("  'm'        : Show model info")
        print("  SPACE      : Pause/Resume")
        print("="*60)
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot access webcam!")
            return
        
        # Webcam settings for better quality
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        
        # State variables
        save_counter = 1
        show_features = True
        fullscreen = False
        paused = False
        analysis_mode = False
        show_model_info = False
        
        # Create output directories
        output_dirs = ['webcam_predictions', 'feature_analysis', 'validation_tests']
        for dir_name in output_dirs:
            os.makedirs(dir_name, exist_ok=True)
        
        print("🎯 Webcam initialized! Position fruit in the center rectangle...")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from webcam")
                    break
            
            # Define ROI (Region of Interest)
            h, w, _ = frame.shape
            roi_size = min(h, w) // 3
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            
            roi = frame[y1:y2, x1:x2]
            
            # Make prediction
            prediction, confidence, selected_features, all_features = self.predict_ripeness(roi)
            
            # Smooth predictions untuk stabilitas
            if prediction != "Error":
                smooth_pred, smooth_conf = self.smooth_predictions(prediction, confidence)
            else:
                smooth_pred, smooth_conf = "Error", 0.0
            
            # Create visualization overlay
            overlay = frame.copy()
            
            # ROI rectangle dengan warna berdasarkan prediksi
            if smooth_pred == 'matang':
                roi_color = (0, 255, 0)      # Green
                bg_color = (0, 60, 0)
                text_color = (255, 255, 255)
            elif smooth_pred == 'mentah':
                roi_color = (0, 165, 255)    # Orange
                bg_color = (0, 40, 60)
                text_color = (255, 255, 255)
            else:
                roi_color = (0, 0, 255)      # Red
                bg_color = (0, 0, 60)
                text_color = (255, 255, 255)
            
            # Draw ROI rectangle
            cv2.rectangle(overlay, (x1, y1), (x2, y2), roi_color, 4)
            
            # Corner markers untuk better visibility
            corner_size = 20
            corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
            for corner in corners:
                cv2.circle(overlay, corner, 8, roi_color, -1)
            
            # Main prediction display
            pred_text = f"PREDIKSI: {smooth_pred.upper()}"
            conf_text = f"CONFIDENCE: {smooth_conf:.1%}"
            raw_text = f"Raw: {prediction} ({confidence:.2f})"
            
            # Text background
            text_w, text_h = 350, 90
            text_x, text_y = x1, y1 - text_h - 10
            cv2.rectangle(overlay, (text_x, text_y), (text_x + text_w, text_y + text_h), bg_color, -1)
            cv2.rectangle(overlay, (text_x, text_y), (text_x + text_w, text_y + text_h), roi_color, 2)
            
            # Draw prediction text
            cv2.putText(overlay, pred_text, (text_x + 10, text_y + 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
            cv2.putText(overlay, conf_text, (text_x + 10, text_y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            cv2.putText(overlay, raw_text, (text_x + 10, text_y + 75),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Feature display
            if show_features and prediction != "Error":
                self.display_features(overlay, selected_features, all_features, analysis_mode)
            
            # Model info display
            if show_model_info:
                self.display_model_info(overlay, w)
            
            # Model info header
            model_info = f"Model: {self.model_name} | Acc: {self.model_results.get('test_accuracy', 0.0):.1%} | Features: {len(self.selected_features)}"
            cv2.putText(overlay, model_info, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # History and status indicators
            history_text = f"History: {len(self.prediction_history)}/10"
            cv2.putText(overlay, history_text, (w - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Analysis mode indicator
            if analysis_mode:
                cv2.putText(overlay, "ANALYSIS MODE", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            # Model info indicator
            if show_model_info:
                cv2.putText(overlay, "MODEL INFO", (10, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 255), 2)
            
            # Pause indicator
            if paused:
                pause_text = "PAUSED - Press SPACE to resume"
                text_size = cv2.getTextSize(pause_text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                pause_x = (w - text_size[0]) // 2
                pause_y = h // 2
                cv2.rectangle(overlay, (pause_x - 20, pause_y - 40), 
                             (pause_x + text_size[0] + 20, pause_y + 20), (0, 0, 0), -1)
                cv2.putText(overlay, pause_text, (pause_x, pause_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
            
            # Status bar
            self.display_status_bar(overlay, w, h, smooth_pred, smooth_conf)
            
            # Display frame
            window_name = "Fruit Ripeness Classifier - Improved Model"
            if fullscreen:
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 1200, 800)
            
            cv2.imshow(window_name, overlay)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                break
            elif key == ord('s') and prediction != "Error":  # Save
                try:
                    self.save_prediction(roi, overlay, prediction, confidence, 
                                       smooth_pred, smooth_conf, selected_features, all_features, 
                                       save_counter, analysis_mode)
                    save_counter += 1
                except Exception as save_error:
                    print(f"❌ Save failed: {save_error}")
                    print("📝 Continuing with classification...")
            elif key == ord('r'):  # Reset history
                self.prediction_history = []
                print("🔄 Prediction history reset")
            elif key == ord('h'):  # Toggle features
                show_features = not show_features
                print(f"📊 Feature display: {'ON' if show_features else 'OFF'}")
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = not fullscreen
                print(f"🖥️  Fullscreen: {'ON' if fullscreen else 'OFF'}")
            elif key == ord('a'):  # Toggle analysis mode
                analysis_mode = not analysis_mode
                print(f"🔬 Analysis mode: {'ON' if analysis_mode else 'OFF'}")
            elif key == ord('m'):  # Toggle model info
                show_model_info = not show_model_info
                print(f"ℹ️  Model info: {'ON' if show_model_info else 'OFF'}")
            elif key == ord(' '):  # Pause/Resume
                paused = not paused
                print(f"⏸️  {'Paused' if paused else 'Resumed'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Session summary
        print(f"\n✅ Classification session completed!")
        print(f"📊 Session Statistics:")
        print(f"   Total predictions saved: {save_counter - 1}")
        print(f"   Final history length: {len(self.prediction_history)}")
        
        if len(self.prediction_history) > 0:
            matang_count = sum(1 for p in self.prediction_history if p[0] == 'matang')
            mentah_count = len(self.prediction_history) - matang_count
            avg_confidence = np.mean([p[1] for p in self.prediction_history])
            print(f"   Predictions distribution: {matang_count} matang, {mentah_count} mentah")
            print(f"   Average confidence: {avg_confidence:.1%}")
    
    def display_features(self, overlay, selected_features, all_features, analysis_mode=False):
        """Display feature values dengan mode normal atau analysis"""
        if analysis_mode:
            # Detailed analysis mode - show selected features dengan progress bars
            bg_w, bg_h = 500, min(len(self.selected_features) * 25 + 60, 400)
            cv2.rectangle(overlay, (10, 80), (10 + bg_w, 80 + bg_h), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 80), (10 + bg_w, 80 + bg_h), (100, 100, 100), 2)
            cv2.putText(overlay, "SELECTED FEATURES (ANALYSIS MODE):", (20, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for i, (feature_name, value) in enumerate(zip(self.selected_features, selected_features)):
                # Determine expected ranges (simplified)
                if 'Mean H' in feature_name:
                    min_val, max_val = 0, 180
                    unit = "°"
                elif 'Mean S' in feature_name or 'Mean V' in feature_name:
                    min_val, max_val = 0, 255
                    unit = ""
                elif 'Std' in feature_name:
                    min_val, max_val = 0, 100
                    unit = ""
                elif 'Ratio' in feature_name:
                    min_val, max_val = 0, 5
                    unit = ""
                elif 'Prop' in feature_name:
                    min_val, max_val = 0, 1
                    unit = "%"
                elif 'Entropy' in feature_name:
                    min_val, max_val = 0, 8
                    unit = ""
                else:
                    min_val, max_val = 0, 100
                    unit = ""
                
                # Normalize value for progress bar
                norm_val = (value - min_val) / (max_val - min_val) if max_val > min_val else 0.5
                norm_val = max(0, min(1, norm_val))
                
                # Color coding
                if norm_val > 0.7:
                    color = (0, 255, 0)  # Green (high)
                elif norm_val > 0.3:
                    color = (0, 255, 255)  # Yellow (medium)
                else:
                    color = (255, 0, 0)  # Blue (low)
                
                # Format text
                if unit == "%":
                    text = f"{feature_name[:12]:>12}: {value*100:6.1f}{unit}"
                elif unit == "°":
                    text = f"{feature_name[:12]:>12}: {value:6.0f}{unit}"
                else:
                    text = f"{feature_name[:12]:>12}: {value:6.2f}{unit}"
                
                y_pos = 130 + i * 22
                if y_pos < 80 + bg_h - 30:  # Ensure it fits in the box
                    cv2.putText(overlay, text, (20, y_pos),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                    
                    # Progress bar
                    bar_x = 250
                    bar_w = 200
                    bar_h = 8
                    cv2.rectangle(overlay, (bar_x, y_pos - 10), 
                                 (bar_x + bar_w, y_pos - 2), (50, 50, 50), -1)
                    cv2.rectangle(overlay, (bar_x, y_pos - 10), 
                                 (bar_x + int(bar_w * norm_val), y_pos - 2), color, -1)
        else:
            # Simple mode - show key selected features only
            key_indices = [i for i, name in enumerate(self.selected_features) 
                          if any(key in name for key in ['Mean H', 'Mean S', 'Mean V', 'Prop'])]
            
            if len(key_indices) > 6:
                key_indices = key_indices[:6]  # Show max 6 features
            
            bg_w, bg_h = 350, len(key_indices) * 22 + 60
            cv2.rectangle(overlay, (10, 80), (10 + bg_w, 80 + bg_h), (0, 0, 0), -1)
            cv2.rectangle(overlay, (10, 80), (10 + bg_w, 80 + bg_h), (100, 100, 100), 2)
            cv2.putText(overlay, "KEY SELECTED FEATURES:", (20, 105), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            for display_i, actual_i in enumerate(key_indices):
                feature_name = self.selected_features[actual_i]
                value = selected_features[actual_i]
                
                if 'Prop' in feature_name:
                    text = f"{feature_name[:12]:>12}: {value*100:6.1f}%"
                elif 'Mean H' in feature_name:
                    text = f"{feature_name[:12]:>12}: {value:6.0f}°"
                else:
                    text = f"{feature_name[:12]:>12}: {value:6.1f}"
                
                cv2.putText(overlay, text, (20, 130 + display_i*22),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def display_model_info(self, overlay, w):
        """Display detailed model information"""
        info_w, info_h = 400, 300
        info_x = w - info_w - 10
        info_y = 60
        
        # Background
        cv2.rectangle(overlay, (info_x, info_y), (info_x + info_w, info_y + info_h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (info_x, info_y), (info_x + info_w, info_y + info_h), (100, 100, 100), 2)
        
        # Title
        cv2.putText(overlay, "MODEL INFORMATION:", (info_x + 10, info_y + 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Model details
        info_lines = [
            f"Model Type: {self.model_name}",
            f"Test Accuracy: {self.model_results.get('test_accuracy', 0.0):.1%}",
            f"F1 Score: {self.model_results.get('f1_score', 0.0):.1%}",
            f"Precision: {self.model_results.get('precision', 0.0):.1%}",
            f"Recall: {self.model_results.get('recall', 0.0):.1%}",
            f"AUC Score: {self.model_results.get('auc_score', 0.0):.1%}",
            "",
            f"Total Features: {len(self.all_features)}",
            f"Selected Features: {len(self.selected_features)}",
            f"Feature Selection: {'Yes' if self.feature_selector else 'No'}",
            f"Scaling: {'RobustScaler' if self.scaler else 'None'}",
            "",
            "Selected Features:",
        ]
        
        # Add selected feature names (first few)
        for i, feature_name in enumerate(self.selected_features[:5]):
            info_lines.append(f"  {i+1}. {feature_name}")
        
        if len(self.selected_features) > 5:
            info_lines.append(f"  ... and {len(self.selected_features)-5} more")
        
        # Draw info lines
        for i, line in enumerate(info_lines):
            if i * 18 + 50 < info_h - 20:  # Ensure it fits
                color = (200, 200, 200) if line.strip() else (100, 100, 100)
                cv2.putText(overlay, line, (info_x + 10, info_y + 50 + i*18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def display_status_bar(self, overlay, w, h, prediction, confidence):
        """Display status bar di bagian bawah"""
        status_h = 60
        status_y = h - status_h
        
        # Background
        cv2.rectangle(overlay, (0, status_y), (w, h), (0, 0, 0), -1)
        cv2.rectangle(overlay, (0, status_y), (w, status_y + 2), (100, 100, 100), -1)
        
        # Current time
        current_time = datetime.now().strftime("%H:%M:%S")
        cv2.putText(overlay, f"Time: {current_time}", (10, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Controls
        controls = "Controls: 'q'=quit | 's'=save | 'r'=reset | 'h'=features | 'f'=fullscreen | 'a'=analysis | 'm'=model | SPACE=pause"
        cv2.putText(overlay, controls, (10, status_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Current prediction summary
        pred_summary = f"Current: {prediction} ({confidence:.1%})"
        cv2.putText(overlay, pred_summary, (w - 250, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Model summary
        model_summary = f"{self.model_name} | {len(self.selected_features)} features"
        cv2.putText(overlay, model_summary, (w - 250, status_y + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    
    def save_prediction(self, roi, overlay, prediction, confidence, 
                       smooth_pred, smooth_conf, selected_features, all_features,
                       counter, analysis_mode):
        """Save prediction dengan detail komprehensif dari improved model"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print(f"💾 Saving prediction {counter:03d}...")
            
            # Save ROI image
            roi_path = f"webcam_predictions/roi_{counter:03d}_{timestamp}.jpg"
            cv2.imwrite(roi_path, roi)
            print(f"  ✅ ROI saved: {roi_path}")
            
            # Save full frame
            frame_path = f"webcam_predictions/frame_{counter:03d}_{timestamp}.jpg"
            cv2.imwrite(frame_path, overlay)
            print(f"  ✅ Frame saved: {frame_path}")
            
            # Save feature analysis visualization
            analysis_path = f"feature_analysis/analysis_{counter:03d}_{timestamp}.jpg"
            try:
                self.save_feature_visualization(roi, selected_features, all_features, analysis_path)
            except Exception as viz_error:
                print(f"  ⚠️  Feature visualization failed: {viz_error}")
            
            # Save detailed report
            report_path = f"webcam_predictions/report_{counter:03d}_{timestamp}.txt"
            try:
                self.save_detailed_report(report_path, counter, timestamp, prediction, confidence,
                                         smooth_pred, smooth_conf, selected_features, all_features, analysis_mode)
                print(f"  ✅ Report saved: {report_path}")
            except Exception as report_error:
                print(f"  ⚠️  Report save failed: {report_error}")
            
            print(f"✅ Prediction {counter:03d} saved: {smooth_pred} (confidence: {smooth_conf:.1%})")
            
        except Exception as e:
            print(f"❌ Error saving prediction: {e}")
            print("📝 Continuing with webcam classification...")
    
    def save_feature_visualization(self, roi, selected_features, all_features, save_path):
        """Create comprehensive feature visualization from improved model"""
        try:
            fig, axes = plt.subplots(3, 4, figsize=(16, 12))
            fig.suptitle('Improved Model - Fruit Feature Analysis', fontsize=16, fontweight='bold')
            
            # Original ROI
            axes[0,0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            axes[0,0].set_title('Original ROI')
            axes[0,0].axis('off')
            
            # HSV channels
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            
            axes[0,1].imshow(hsv[:,:,0], cmap='hsv')
            axes[0,1].set_title(f'Hue Channel')
            axes[0,1].axis('off')
            
            axes[0,2].imshow(hsv[:,:,1], cmap='viridis')
            axes[0,2].set_title(f'Saturation Channel')
            axes[0,2].axis('off')
            
            axes[0,3].imshow(hsv[:,:,2], cmap='gray')
            axes[0,3].set_title(f'Value Channel')
            axes[0,3].axis('off')
            
            # Color analysis
            h = hsv[:,:,0]
            yellow_mask = (h >= 20) & (h <= 40)
            green_mask = (h >= 50) & (h <= 70)
            
            # Color masks - using valid colormaps
            axes[1,0].imshow(yellow_mask, cmap='YlOrBr')  # Valid yellow colormap
            axes[1,0].set_title(f'Yellow Regions')
            axes[1,0].axis('off')
            
            axes[1,1].imshow(green_mask, cmap='Greens')  # Valid green colormap
            axes[1,1].set_title(f'Green Regions')
            axes[1,1].axis('off')
            
            # Combined color visualization
            color_viz = np.zeros_like(roi)
            color_viz[yellow_mask] = [0, 255, 255]  # Yellow
            color_viz[green_mask] = [0, 255, 0]     # Green
            
            axes[1,2].imshow(cv2.cvtColor(color_viz, cv2.COLOR_BGR2RGB))
            axes[1,2].set_title('Color Classification')
            axes[1,2].axis('off')
            
            # Texture analysis
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            axes[1,3].imshow(gray, cmap='gray')
            axes[1,3].set_title(f'Texture Analysis')
            axes[1,3].axis('off')
            
            # Selected features visualization
            if len(selected_features) > 0:
                axes[2,0].bar(range(len(selected_features)), selected_features, color='skyblue')
                axes[2,0].set_title('Selected Features')
                axes[2,0].set_xticks(range(len(selected_features)))
                feature_labels = [name[:6] for name in self.selected_features] if self.selected_features else [f'F{i}' for i in range(len(selected_features))]
                axes[2,0].set_xticklabels(feature_labels, rotation=45, fontsize=8)
            else:
                axes[2,0].text(0.5, 0.5, 'No Selected\nFeatures', ha='center', va='center', transform=axes[2,0].transAxes)
                axes[2,0].set_title('Selected Features')
            
            # All features vs selected
            if len(all_features) > 0:
                all_indices = range(len(all_features))
                selected_indices = []
                
                # Map selected features to all features
                if self.selected_features and hasattr(self.feature_extractor, 'feature_names'):
                    feature_dict = dict(zip(self.feature_extractor.feature_names, all_features))
                    for selected_name in self.selected_features:
                        if selected_name in self.feature_extractor.feature_names:
                            idx = self.feature_extractor.feature_names.index(selected_name)
                            selected_indices.append(idx)
                
                colors = ['red' if i in selected_indices else 'lightgray' for i in all_indices]
                axes[2,1].bar(all_indices, all_features, color=colors)
                axes[2,1].set_title('All Features (Red = Selected)')
                axes[2,1].set_xticks(all_indices)
                if hasattr(self.feature_extractor, 'feature_names'):
                    all_labels = [name[:4] for name in self.feature_extractor.feature_names]
                else:
                    all_labels = [f'F{i}' for i in all_indices]
                axes[2,1].set_xticklabels(all_labels, rotation=45, fontsize=6)
            else:
                axes[2,1].text(0.5, 0.5, 'No Features\nExtracted', ha='center', va='center', transform=axes[2,1].transAxes)
                axes[2,1].set_title('All Features')
            
            # Model performance summary
            axes[2,2].text(0.1, 0.9, f'Model Performance:', fontsize=12, fontweight='bold', 
                          transform=axes[2,2].transAxes)
            axes[2,2].text(0.1, 0.8, f'Model: {self.model_name}', transform=axes[2,2].transAxes)
            axes[2,2].text(0.1, 0.7, f'Test Acc: {self.model_results.get("test_accuracy", 0.0):.1%}', 
                          transform=axes[2,2].transAxes)
            axes[2,2].text(0.1, 0.6, f'F1 Score: {self.model_results.get("f1_score", 0.0):.1%}', 
                          transform=axes[2,2].transAxes)
            axes[2,2].text(0.1, 0.5, f'AUC Score: {self.model_results.get("auc_score", 0.0):.1%}', 
                          transform=axes[2,2].transAxes)
            axes[2,2].text(0.1, 0.4, f'Selected: {len(self.selected_features) if self.selected_features else 0}/{len(self.all_features) if self.all_features else 0}', 
                          transform=axes[2,2].transAxes)
            axes[2,2].text(0.1, 0.3, f'Scaling: {"Yes" if self.scaler else "No"}', 
                          transform=axes[2,2].transAxes)
            axes[2,2].text(0.1, 0.2, f'Selection: {"Yes" if self.feature_selector else "No"}', 
                          transform=axes[2,2].transAxes)
            axes[2,2].set_xlim(0, 1)
            axes[2,2].set_ylim(0, 1)
            axes[2,2].axis('off')
            
            # Feature importance or profile
            try:
                if hasattr(self.model, 'feature_importances_') and len(selected_features) > 0:
                    importances = self.model.feature_importances_
                    if len(importances) == len(selected_features):
                        axes[2,3].bar(range(len(importances)), importances, color='green')
                        axes[2,3].set_title('Feature Importance')
                        axes[2,3].set_xticks(range(len(importances)))
                        feature_labels = [name[:6] for name in self.selected_features] if self.selected_features else [f'F{i}' for i in range(len(importances))]
                        axes[2,3].set_xticklabels(feature_labels, rotation=45, fontsize=8)
                    else:
                        axes[2,3].text(0.5, 0.5, f'Feature Importance\nDimension Mismatch\n({len(importances)} vs {len(selected_features)})', 
                                      ha='center', va='center', transform=axes[2,3].transAxes)
                        axes[2,3].axis('off')
                elif len(selected_features) >= 3:
                    # Normalized selected features radar plot
                    angles = np.linspace(0, 2*np.pi, len(selected_features), endpoint=False)
                    norm_features = selected_features / (np.max(selected_features) + 1e-8)
                    
                    axes[2,3] = plt.subplot(3, 4, 12, projection='polar')
                    axes[2,3].plot(angles, norm_features, 'o-', linewidth=2, color='purple')
                    axes[2,3].fill(angles, norm_features, alpha=0.25, color='purple')
                    axes[2,3].set_title('Feature Profile')
                    axes[2,3].set_xticks(angles)
                    feature_labels = [name[:4] for name in self.selected_features] if self.selected_features else [f'F{i}' for i in range(len(selected_features))]
                    axes[2,3].set_xticklabels(feature_labels, fontsize=6)
                else:
                    axes[2,3].text(0.5, 0.5, 'Feature Profile\nNot Available\n(< 3 features)', 
                                  ha='center', va='center', transform=axes[2,3].transAxes)
                    axes[2,3].axis('off')
            except Exception as plot_error:
                axes[2,3].text(0.5, 0.5, f'Feature Analysis\nError:\n{str(plot_error)[:30]}...', 
                              ha='center', va='center', transform=axes[2,3].transAxes)
                axes[2,3].axis('off')
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✅ Feature visualization saved: {save_path}")
            
        except Exception as e:
            print(f"⚠️  Error creating feature visualization: {e}")
            # Create a simple fallback visualization
            try:
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                ax.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                ax.set_title('ROI Analysis (Simplified)')
                ax.axis('off')
                
                # Add text info
                info_text = f"Model: {self.model_name}\n"
                info_text += f"Features: {len(selected_features)} selected\n"
                info_text += f"Accuracy: {self.model_results.get('test_accuracy', 0.0):.1%}"
                
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✅ Fallback visualization saved: {save_path}")
            except Exception as fallback_error:
                print(f"❌ Failed to create fallback visualization: {fallback_error}")
    
    def save_detailed_report(self, report_path, counter, timestamp, prediction, confidence,
                           smooth_pred, smooth_conf, selected_features, all_features, analysis_mode):
        """Save detailed prediction report from improved model"""
        with open(report_path, 'w') as f:
            f.write("🍎 IMPROVED MODEL - FRUIT RIPENESS PREDICTION REPORT\n")
            f.write("="*60 + "\n\n")
            
            # Basic info
            f.write(f"📊 PREDICTION SUMMARY\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Counter: {counter:03d}\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Analysis Mode: {'ON' if analysis_mode else 'OFF'}\n\n")
            
            # Model performance
            f.write(f"🏆 MODEL PERFORMANCE\n")
            f.write(f"Test Accuracy: {self.model_results.get('test_accuracy', 0.0):.4f} ({self.model_results.get('test_accuracy', 0.0)*100:.1f}%)\n")
            f.write(f"F1 Score: {self.model_results.get('f1_score', 0.0):.4f} ({self.model_results.get('f1_score', 0.0)*100:.1f}%)\n")
            f.write(f"Precision: {self.model_results.get('precision', 0.0):.4f} ({self.model_results.get('precision', 0.0)*100:.1f}%)\n")
            f.write(f"Recall: {self.model_results.get('recall', 0.0):.4f} ({self.model_results.get('recall', 0.0)*100:.1f}%)\n")
            f.write(f"AUC Score: {self.model_results.get('auc_score', 0.0):.4f} ({self.model_results.get('auc_score', 0.0)*100:.1f}%)\n\n")
            
            # Predictions
            f.write(f"🎯 PREDICTION RESULTS\n")
            f.write(f"Raw Prediction: {prediction}\n")
            f.write(f"Raw Confidence: {confidence:.4f} ({confidence*100:.1f}%)\n")
            f.write(f"Smoothed Prediction: {smooth_pred}\n")
            f.write(f"Smoothed Confidence: {smooth_conf:.4f} ({smooth_conf*100:.1f}%)\n")
            f.write(f"History Length: {len(self.prediction_history)}\n\n")
            
            # Feature engineering details
            f.write(f"🔧 FEATURE ENGINEERING\n")
            f.write(f"Total Available Features: {len(self.all_features)}\n")
            f.write(f"Selected Features: {len(self.selected_features)}\n")
            f.write(f"Feature Selection Method: {'SelectKBest' if self.feature_selector else 'Manual'}\n")
            f.write(f"Scaling Method: {'RobustScaler' if self.scaler else 'None'}\n\n")
            
            # Selected feature details
            f.write(f"🔬 SELECTED FEATURES\n")
            f.write("-" * 40 + "\n")
            for i, name in enumerate(self.selected_features):
                value = selected_features[i] if i < len(selected_features) else 0.0
                if 'Prop' in name:
                    f.write(f"{name:>18}: {value:8.4f} ({value*100:5.1f}%)\n")
                elif 'H' in name and ('Mean' in name or 'Std' in name):
                    f.write(f"{name:>18}: {value:8.1f}°\n")
                else:
                    f.write(f"{name:>18}: {value:8.3f}\n")
            
            # All features (for reference)
            f.write(f"\n📋 ALL AVAILABLE FEATURES (REFERENCE)\n")
            f.write("-" * 40 + "\n")
            for i, name in enumerate(self.feature_extractor.feature_names):
                value = all_features[i] if i < len(all_features) else 0.0
                selected = "✓" if name in self.selected_features else " "
                if 'Prop' in name:
                    f.write(f"{selected} {name:>15}: {value:8.4f} ({value*100:5.1f}%)\n")
                elif 'H' in name and ('Mean' in name or 'Std' in name):
                    f.write(f"{selected} {name:>15}: {value:8.1f}°\n")
                else:
                    f.write(f"{selected} {name:>15}: {value:8.3f}\n")
            
            # Feature analysis
            f.write(f"\n📈 FEATURE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            
            # Analyze key features
            feature_dict = dict(zip(self.selected_features, selected_features))
            
            f.write(f"Color Analysis:\n")
            if 'Mean H' in feature_dict:
                hue_val = feature_dict['Mean H']
                f.write(f"  Dominant Hue: {hue_val:.1f}° ")
                if hue_val < 30:
                    f.write("(Red/Orange spectrum)\n")
                elif hue_val < 60:
                    f.write("(Yellow spectrum)\n")
                elif hue_val < 120:
                    f.write("(Green spectrum)\n")
                else:
                    f.write("(Blue/Purple spectrum)\n")
            
            if 'Mean S' in feature_dict:
                sat_val = feature_dict['Mean S']
                f.write(f"  Saturation Level: {sat_val:.1f} ")
                if sat_val > 200:
                    f.write("(Very saturated)\n")
                elif sat_val > 100:
                    f.write("(Moderately saturated)\n")
                else:
                    f.write("(Low saturation)\n")
            
            if 'Prop Kuning' in feature_dict:
                f.write(f"  Yellow Content: {feature_dict['Prop Kuning']*100:.1f}%\n")
            
            if 'Prop Hijau' in feature_dict:
                f.write(f"  Green Content: {feature_dict['Prop Hijau']*100:.1f}%\n")
            
            if 'Std Gray' in feature_dict:
                texture_val = feature_dict['Std Gray']
                f.write(f"\nTexture Analysis:\n")
                f.write(f"  Grayscale Variation: {texture_val:.1f} ")
                if texture_val > 50:
                    f.write("(High texture variation)\n")
                elif texture_val > 30:
                    f.write("(Moderate texture)\n")
                else:
                    f.write("(Smooth texture)\n")
            
            # Prediction confidence analysis
            f.write(f"\n🎯 CONFIDENCE ANALYSIS\n")
            f.write("-" * 40 + "\n")
            if smooth_conf > 0.8:
                f.write("High confidence prediction - Very reliable\n")
                f.write("The model is very certain about this classification.\n")
            elif smooth_conf > 0.6:
                f.write("Medium confidence prediction - Fairly reliable\n")
                f.write("The model has reasonable certainty about this classification.\n")
            else:
                f.write("Low confidence prediction - May need manual verification\n")
                f.write("The model is uncertain. Consider manual inspection.\n")
            
            # Model comparison note
            f.write(f"\n📊 MODEL NOTES\n")
            f.write("-" * 40 + "\n")
            f.write(f"This prediction was made using the improved ML pipeline.\n")
            f.write(f"Key improvements over basic model:\n")
            f.write(f"- Advanced feature selection (SelectKBest)\n")
            f.write(f"- Robust scaling for outlier handling\n")
            f.write(f"- Cross-validation for model selection\n")
            f.write(f"- Hyperparameter tuning\n")
            f.write(f"- Multiple evaluation metrics\n")
            
            # Files generated
            f.write(f"\n📁 GENERATED FILES\n")
            f.write("-" * 40 + "\n")
            f.write(f"ROI Image: webcam_predictions/roi_{counter:03d}_{timestamp}.jpg\n")
            f.write(f"Full Frame: webcam_predictions/frame_{counter:03d}_{timestamp}.jpg\n")
            f.write(f"Feature Analysis: feature_analysis/analysis_{counter:03d}_{timestamp}.jpg\n")

def main():
    """Main function untuk menjalankan improved webcam classifier"""
    print("🍎" * 30)
    print("IMPROVED FRUIT RIPENESS WEBCAM CLASSIFIER")
    print("🍎" * 30)
    print("\nLoading improved pre-trained model...")
    
    # List of possible model files to try
    model_files = [
        'fruit_ripeness_model_improved.pkl',
        'fruit_ripeness_model.pkl',  # Fallback to basic model
    ]
    
    classifier = None
    
    for model_file in model_files:
        try:
            print(f"\n🔍 Trying model file: {model_file}")
            classifier = WebcamFruitClassifier(model_file)
            break
        except FileNotFoundError:
            print(f"❌ {model_file} not found")
            continue
        except Exception as e:
            print(f"❌ Failed to load {model_file}: {e}")
            continue
    
    if classifier is None:
        print("\n" + "❌" * 40)
        print("NO COMPATIBLE MODEL FILE FOUND!")
        print("❌" * 40)
        print("\nPlease run one of these training scripts first:")
        print("1. python fruit_classification_improved.py  (Recommended)")
        print("2. python complete_fruit_system.py         (Basic version)")
        print("\nImproved model features:")
        print("- Advanced feature selection")
        print("- Robust scaling")
        print("- Cross-validation")
        print("- Hyperparameter tuning")
        print("- Multiple evaluation metrics")
        print("\nAfter training, you can run this script for webcam-only mode.")
        return
    
    try:
        print("\n⚡ Starting improved real-time classification...")
        classifier.run_classification()
        
    except KeyboardInterrupt:
        print("\n⏹️  Classification interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Unexpected error during classification: {e}")
        print("\nTroubleshooting:")
        print("1. Check if your webcam is connected and working")
        print("2. Ensure sufficient lighting for fruit detection")
        print("3. Verify all required libraries are installed")
        print("4. Try running: pip install opencv-python matplotlib numpy scikit-learn joblib")
        print("5. Check model compatibility - ensure you're using the improved model")
        print("6. Restart the application if webcam issues persist")

if __name__ == "__main__":
    main()