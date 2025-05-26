# ==============================================================================
# FRUIT RIPENESS CLASSIFICATION SYSTEM - IMPROVED VERSION
# Sistem Klasifikasi Kematangan Buah dengan Validasi File Gambar
# ==============================================================================

# =============================================================================
# 1. IMPORT LIBRARIES
# =============================================================================

# Core Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import os
import warnings
import joblib
from datetime import datetime
from pathlib import Path
import requests
from urllib.parse import urlparse
import time
import re
from collections import Counter

warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA

# Deep Learning Libraries (Image Processing Focus)
try:
    import tensorflow as tf
    from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
    from tensorflow.keras.models import Sequential, Model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
    from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.preprocessing.image import img_to_array
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("TensorFlow not available. Using only traditional ML models.")

# Statistical Analysis
from scipy import stats
from scipy.stats import chi2_contingency

# =============================================================================
# 2. DATA CLEANING UTILITIES
# =============================================================================

class DataCleaner:
    """
    Data cleaning utilities untuk menangani format data yang bermasalah
    """
    
    @staticmethod
    def clean_numeric_string(value):
        """Clean numeric string dengan format Eropa (titik sebagai pemisah ribuan)"""
        if pd.isna(value) or value == '':
            return np.nan
        
        # Convert to string
        value_str = str(value)
        
        # Handle scientific notation
        if 'e' in value_str.lower() or 'E' in value_str:
            try:
                return float(value_str)
            except:
                return np.nan
        
        # Remove any non-numeric characters except dots, commas, and minus
        cleaned = re.sub(r'[^\d.,-]', '', value_str)
        
        # Count dots and commas
        dot_count = cleaned.count('.')
        comma_count = cleaned.count(',')
        
        try:
            # Case 1: European format with dots as thousand separators and comma as decimal
            if dot_count > 1 or (dot_count >= 1 and comma_count == 1 and cleaned.rfind(',') > cleaned.rfind('.')):
                cleaned = cleaned.replace('.', '')
                cleaned = cleaned.replace(',', '.')
                return float(cleaned)
            
            # Case 2: US format with commas as thousand separators and dot as decimal
            elif comma_count > 1 or (comma_count >= 1 and dot_count == 1 and cleaned.rfind('.') > cleaned.rfind(',')):
                cleaned = cleaned.replace(',', '')
                return float(cleaned)
            
            # Case 3: Simple decimal with dot
            elif dot_count == 1 and comma_count == 0:
                return float(cleaned)
            
            # Case 4: Simple decimal with comma (European)
            elif comma_count == 1 and dot_count == 0:
                cleaned = cleaned.replace(',', '.')
                return float(cleaned)
            
            # Case 5: Integer without separators
            elif dot_count == 0 and comma_count == 0:
                return float(cleaned)
            
            # Case 6: Multiple dots without comma (European thousand separators)
            elif dot_count > 1 and comma_count == 0:
                parts = cleaned.split('.')
                if len(parts[-1]) <= 3:
                    cleaned = cleaned.replace('.', '')
                    return float(cleaned)
                else:
                    main_part = '.'.join(parts[:-1]).replace('.', '')
                    decimal_part = parts[-1]
                    cleaned = main_part + '.' + decimal_part
                    return float(cleaned)
            
            else:
                return float(cleaned)
                
        except ValueError:
            print(f"Warning: Could not convert '{value}' to float")
            return np.nan
    
    @staticmethod
    def clean_dataframe(df):
        """Clean entire dataframe numeric columns"""
        print("🧹 Cleaning numeric data...")
        
        # Identify numeric columns (exclude Label and Filename)
        numeric_cols = [col for col in df.columns if col not in ['Label', 'Filename']]
        
        # Clean each numeric column
        for col in numeric_cols:
            print(f"  Cleaning column: {col}")
            original_values = df[col].copy()
            
            # Apply cleaning function
            df[col] = df[col].apply(DataCleaner.clean_numeric_string)
            
            # Check for any remaining non-numeric values
            non_numeric_mask = pd.isna(df[col]) & pd.notna(original_values)
            if non_numeric_mask.any():
                problematic_values = original_values[non_numeric_mask].unique()
                print(f"    Warning: Could not clean {len(problematic_values)} unique values in {col}")
                print(f"    Examples: {list(problematic_values[:3])}")
        
        # Remove rows with too many NaN values (more than 50% of numeric columns)
        nan_threshold = len(numeric_cols) * 0.5
        rows_with_many_nans = df[numeric_cols].isna().sum(axis=1) > nan_threshold
        
        if rows_with_many_nans.any():
            print(f"  Removing {rows_with_many_nans.sum()} rows with excessive missing values")
            df = df[~rows_with_many_nans].copy()
        
        # Fill remaining NaN values with column median
        for col in numeric_cols:
            nan_count = df[col].isna().sum()
            if nan_count > 0:
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                print(f"  Filled {nan_count} NaN values in {col} with median: {median_val:.3f}")
        
        print(f"✅ Data cleaning completed. Final shape: {df.shape}")
        return df

# =============================================================================
# 3. FEATURE EXTRACTION
# =============================================================================

class FruitFeatureExtractor:
    """
    Feature Extractor untuk ekstraksi fitur handcrafted dari citra buah
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
# 4. DATASET MANAGER
# =============================================================================

class FruitDatasetManager:
    """
    Manager untuk loading dan preprocessing dataset buah
    """
    
    def __init__(self, csv_url, images_folder="dataset_images"):
        self.csv_url = csv_url
        self.images_folder = Path(images_folder)
        self.df = None
        self.feature_extractor = FruitFeatureExtractor()
        
        # Create images folder
        self.images_folder.mkdir(exist_ok=True)
    
    def load_dataset(self):
        """Load dataset dari Google Sheets"""
        try:
            # Convert Google Sheets URL to CSV export URL
            if "docs.google.com/spreadsheets" in self.csv_url:
                sheet_id = self.csv_url.split('/d/')[1].split('/')[0]
                csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
            else:
                csv_url = self.csv_url
            
            print("Loading dataset from Google Sheets...")
            self.df = pd.read_csv(csv_url)
            print(f"Dataset loaded! Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            
            # Check if Filename column exists
            if 'Filename' not in self.df.columns:
                print("Warning: 'Filename' column not found!")
                return False
            
            # Clean the data
            self.df = DataCleaner.clean_dataframe(self.df)
            
            # Filter out 'busuk' class if exists (focus on matang vs mentah)
            if 'busuk' in self.df['Label'].values:
                print("🗑️  Removing 'busuk' class for binary classification (matang vs mentah)")
                original_shape = self.df.shape
                self.df = self.df[self.df['Label'] != 'busuk'].copy()
                print(f"   Shape after filtering: {self.df.shape} (removed {original_shape[0] - self.df.shape[0]} rows)")
            
            # Reset index after filtering
            self.df = self.df.reset_index(drop=True)
            
            print(f"📊 Final dataset info:")
            print(f"   Shape: {self.df.shape}")
            print(f"   Classes: {dict(self.df['Label'].value_counts())}")
            
            return True
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Creating sample dataset as fallback...")
            return self.create_sample_dataset()
    
    def create_sample_dataset(self):
        """Create sample dataset jika loading gagal"""
        print("Creating sample dataset...")
        np.random.seed(42)
        
        n_samples = 100
        data = {
            'Mean H': np.random.uniform(20, 120, n_samples),
            'Mean S': np.random.uniform(50, 200, n_samples),
            'Mean V': np.random.uniform(100, 255, n_samples),
            'Std H': np.random.uniform(2, 80, n_samples),
            'Std S': np.random.uniform(20, 150, n_samples),
            'Std V': np.random.uniform(30, 100, n_samples),
            'Ratio S/H': np.random.uniform(0.5, 5, n_samples),
            'Ratio V/S': np.random.uniform(0.8, 3, n_samples),
            'Entropy H': np.random.uniform(3, 6, n_samples),
            'Prop Kuning': np.random.uniform(0, 0.8, n_samples),
            'Prop Hijau': np.random.uniform(0, 0.6, n_samples),
            'Std Gray': np.random.uniform(30, 70, n_samples),
        }
        
        # Create labels with logic
        labels = []
        filenames = []
        for i in range(n_samples):
            if data['Prop Kuning'][i] > 0.3 and data['Mean V'][i] > 180:
                labels.append('matang')
                filenames.append(f'matang-{(i//2) + 1}.jpg')
            else:
                labels.append('mentah')
                filenames.append(f'mentah-{(i//2) + 1}.jpg')
        
        data['Label'] = labels
        data['Filename'] = filenames
        self.df = pd.DataFrame(data)
        
        print(f"Sample dataset created! Shape: {self.df.shape}")
        return True
    
    def download_sample_images(self):
        """Download atau create sample images untuk testing"""
        print("Creating sample images for testing...")
        
        unique_files = self.df['Filename'].unique()
        
        for filename in unique_files:
            image_path = self.images_folder / filename
            
            if not image_path.exists():
                row = self.df[self.df['Filename'] == filename].iloc[0]
                img = self.create_synthetic_image(row)
                cv2.imwrite(str(image_path), img)
        
        print(f"Sample images created in {self.images_folder}")
        print(f"Total unique images: {len(unique_files)}")
    
    def create_synthetic_image(self, row):
        """Create synthetic fruit image based on features"""
        img = np.zeros((200, 200, 3), dtype=np.uint8)
        
        center = (100, 100)
        radius = 80
        
        if row['Label'] == 'matang':
            prop_kuning = min(row.get('Prop Kuning', 0.5), 1.0)
            color = (
                int(30 + prop_kuning * 100),
                int(150 + prop_kuning * 105),
                int(200 - prop_kuning * 50)
            )
        else:
            prop_hijau = min(row.get('Prop Hijau', 0.5), 1.0)
            color = (
                int(50 + prop_hijau * 30),
                int(180 + prop_hijau * 75),
                int(100 - prop_hijau * 50)
            )
        
        cv2.circle(img, center, radius, color, -1)
        
        std_gray = row.get('Std Gray', 40)
        noise_intensity = int(min(std_gray * 0.5, 30))
        noise = np.random.randint(-noise_intensity, noise_intensity, (200, 200, 3))
        img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)
        
        return img

# =============================================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================

class EDAnalyzer:
    """
    Comprehensive Exploratory Data Analysis untuk dataset buah
    """
    
    def __init__(self, df):
        self.df = df
        self.feature_cols = [col for col in df.columns if col not in ['Label', 'Filename']]
        
    def comprehensive_eda(self):
        """Perform comprehensive EDA"""
        print("\n" + "="*60)
        print("📊 COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        
        # 1. Dataset Overview
        self._plot_dataset_overview(fig)
        
        # 2. Class Distribution Analysis
        self._plot_class_distribution(fig)
        
        # 3. Feature Distribution Analysis
        self._plot_feature_distributions(fig)
        
        # 4. Correlation Analysis
        self._plot_correlation_analysis(fig)
        
        # 5. Feature Importance via Statistical Tests
        self._plot_statistical_analysis(fig)
        
        # 6. Outlier Analysis
        self._plot_outlier_analysis(fig)
        
        # 7. Feature Engineering Insights
        self._plot_feature_engineering_insights(fig)
        
        plt.tight_layout()
        plt.savefig('comprehensive_eda_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print statistical summaries
        self._print_statistical_summaries()
    
    def _plot_dataset_overview(self, fig):
        """Plot dataset overview"""
        # Class distribution pie chart
        ax1 = plt.subplot(4, 4, 1)
        class_counts = self.df['Label'].value_counts()
        colors = ['#FF6B6B', '#4ECDC4']
        wedges, texts, autotexts = ax1.pie(class_counts.values, labels=class_counts.index, 
                                          autopct='%1.1f%%', colors=colors)
        ax1.set_title('Class Distribution', fontsize=12, fontweight='bold')
        
        # Dataset info text
        ax2 = plt.subplot(4, 4, 2)
        ax2.axis('off')
        info_text = f"""Dataset Information:
        
Total Samples: {len(self.df)}
Features: {len(self.feature_cols)}
Classes: {len(class_counts)}

Class Balance:
{class_counts['matang']} Matang samples
{class_counts['mentah']} Mentah samples

Balance Ratio: {class_counts.min()/class_counts.max():.2f}"""
        
        ax2.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
        ax2.set_title('Dataset Summary', fontsize=12, fontweight='bold')
    
    def _plot_class_distribution(self, fig):
        """Plot detailed class distribution analysis"""
        # Feature means by class
        ax3 = plt.subplot(4, 4, 3)
        key_features = ['Mean H', 'Mean S', 'Mean V', 'Prop Kuning']
        if all(f in self.df.columns for f in key_features):
            means_by_class = self.df.groupby('Label')[key_features].mean()
            means_by_class.T.plot(kind='bar', ax=ax3, color=['orange', 'green'])
            ax3.set_title('Feature Means by Class')
            ax3.set_ylabel('Feature Value')
            ax3.tick_params(axis='x', rotation=45)
            ax3.legend()
    
    def _plot_feature_distributions(self, fig):
        """Plot feature distributions"""
        # Key feature histograms
        key_features = ['Mean H', 'Prop Kuning', 'Prop Hijau', 'Mean V']
        
        for i, feature in enumerate(key_features):
            if feature in self.df.columns:
                ax = plt.subplot(4, 4, 4 + i)
                for label in self.df['Label'].unique():
                    data = self.df[self.df['Label'] == label][feature]
                    ax.hist(data, alpha=0.7, label=label, bins=15, density=True)
                ax.set_title(f'{feature} Distribution')
                ax.set_xlabel(feature)
                ax.set_ylabel('Density')
                ax.legend()
    
    def _plot_correlation_analysis(self, fig):
        """Plot correlation analysis"""
        # Correlation matrix
        ax8 = plt.subplot(4, 4, 8)
        numeric_df = self.df[self.feature_cols]
        corr_matrix = numeric_df.corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, ax=ax8, fmt='.2f', cbar_kws={'shrink': 0.8})
        ax8.set_title('Feature Correlation Matrix')
        
        # Top correlations
        ax9 = plt.subplot(4, 4, 9)
        # Find highest correlations (excluding diagonal)
        corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_pairs.append((
                    corr_matrix.columns[i], 
                    corr_matrix.columns[j], 
                    abs(corr_matrix.iloc[i, j])
                ))
        
        top_corr = sorted(corr_pairs, key=lambda x: x[2], reverse=True)[:5]
        
        features = [f"{pair[0][:8]}\nvs\n{pair[1][:8]}" for pair in top_corr]
        corr_values = [pair[2] for pair in top_corr]
        
        bars = ax9.bar(range(len(features)), corr_values, color='skyblue')
        ax9.set_xticks(range(len(features)))
        ax9.set_xticklabels(features, rotation=45, ha='right')
        ax9.set_title('Top 5 Feature Correlations')
        ax9.set_ylabel('|Correlation|')
        
        # Add value labels on bars
        for bar, value in zip(bars, corr_values):
            ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.2f}', ha='center', va='bottom')
    
    def _plot_statistical_analysis(self, fig):
        """Plot statistical analysis"""
        # Feature importance via ANOVA F-test
        ax10 = plt.subplot(4, 4, 10)
        
        X = self.df[self.feature_cols]
        y = self.df['Label']
        
        # Calculate F-statistics
        f_stats, p_values = f_classif(X, y)
        
        # Create DataFrame for plotting
        importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'F_Score': f_stats,
            'P_Value': p_values
        }).sort_values('F_Score', ascending=False)
        
        bars = ax10.bar(range(len(importance_df)), importance_df['F_Score'], 
                       color='lightcoral')
        ax10.set_xticks(range(len(importance_df)))
        ax10.set_xticklabels([f[:8] for f in importance_df['Feature']], 
                            rotation=45, ha='right')
        ax10.set_title('Feature Importance (ANOVA F-test)')
        ax10.set_ylabel('F-Score')
        
        # P-value significance
        ax11 = plt.subplot(4, 4, 11)
        significant = importance_df['P_Value'] < 0.05
        colors = ['green' if sig else 'red' for sig in significant]
        
        bars = ax11.bar(range(len(importance_df)), -np.log10(importance_df['P_Value']), 
                       color=colors)
        ax11.axhline(y=-np.log10(0.05), color='black', linestyle='--', 
                    label='p=0.05 threshold')
        ax11.set_xticks(range(len(importance_df)))
        ax11.set_xticklabels([f[:8] for f in importance_df['Feature']], 
                            rotation=45, ha='right')
        ax11.set_title('Feature Statistical Significance')
        ax11.set_ylabel('-log10(p-value)')
        ax11.legend()
    
    def _plot_outlier_analysis(self, fig):
        """Plot outlier analysis"""
        # Box plots for key features
        ax12 = plt.subplot(4, 4, 12)
        key_features = ['Mean H', 'Mean S', 'Mean V', 'Prop Kuning']
        available_features = [f for f in key_features if f in self.df.columns][:4]
        
        if available_features:
            data_to_plot = [self.df[feature] for feature in available_features]
            box_plot = ax12.boxplot(data_to_plot, labels=[f[:8] for f in available_features])
            ax12.set_title('Outlier Detection (Box Plots)')
            ax12.set_ylabel('Feature Values')
            plt.setp(ax12.get_xticklabels(), rotation=45)
        
        # Outlier count per feature
        ax13 = plt.subplot(4, 4, 13)
        outlier_counts = []
        for feature in self.feature_cols:
            Q1 = self.df[feature].quantile(0.25)
            Q3 = self.df[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = ((self.df[feature] < lower_bound) | 
                       (self.df[feature] > upper_bound)).sum()
            outlier_counts.append(outliers)
        
        bars = ax13.bar(range(len(self.feature_cols)), outlier_counts, color='salmon')
        ax13.set_xticks(range(len(self.feature_cols)))
        ax13.set_xticklabels([f[:8] for f in self.feature_cols], rotation=45, ha='right')
        ax13.set_title('Outlier Count per Feature')
        ax13.set_ylabel('Number of Outliers')
    
    def _plot_feature_engineering_insights(self, fig):
        """Plot feature engineering insights"""
        # Feature ranges by class
        ax14 = plt.subplot(4, 4, 14)
        
        # Calculate feature ranges (max - min) for each class
        ranges_by_class = {}
        for label in self.df['Label'].unique():
            class_data = self.df[self.df['Label'] == label]
            ranges = class_data[self.feature_cols].max() - class_data[self.feature_cols].min()
            ranges_by_class[label] = ranges
        
        # Plot first 6 features for readability
        features_to_plot = self.feature_cols[:6]
        x = np.arange(len(features_to_plot))
        width = 0.35
        
        for i, (label, ranges) in enumerate(ranges_by_class.items()):
            ax14.bar(x + i*width, [ranges[f] for f in features_to_plot], 
                    width, label=label, alpha=0.8)
        
        ax14.set_xlabel('Features')
        ax14.set_ylabel('Range (Max - Min)')
        ax14.set_title('Feature Ranges by Class')
        ax14.set_xticks(x + width/2)
        ax14.set_xticklabels([f[:8] for f in features_to_plot], rotation=45)
        ax14.legend()
        
        # Class separability analysis
        ax15 = plt.subplot(4, 4, 15)
        
        # Calculate class separability (distance between means / pooled std)
        separability_scores = []
        for feature in self.feature_cols:
            class_data = {}
            for label in self.df['Label'].unique():
                class_data[label] = self.df[self.df['Label'] == label][feature]
            
            if len(class_data) == 2:
                labels = list(class_data.keys())
                mean_diff = abs(class_data[labels[0]].mean() - class_data[labels[1]].mean())
                pooled_std = np.sqrt((class_data[labels[0]].var() + class_data[labels[1]].var()) / 2)
                separability = mean_diff / (pooled_std + 1e-6)
                separability_scores.append(separability)
            else:
                separability_scores.append(0)
        
        bars = ax15.bar(range(len(self.feature_cols)), separability_scores, color='lightgreen')
        ax15.set_xticks(range(len(self.feature_cols)))
        ax15.set_xticklabels([f[:8] for f in self.feature_cols], rotation=45, ha='right')
        ax15.set_title('Class Separability by Feature')
        ax15.set_ylabel('Separability Score')
        
        # Feature interaction scatter plot
        ax16 = plt.subplot(4, 4, 16)
        if 'Prop Kuning' in self.df.columns and 'Mean V' in self.df.columns:
            for label in self.df['Label'].unique():
                class_data = self.df[self.df['Label'] == label]
                ax16.scatter(class_data['Prop Kuning'], class_data['Mean V'], 
                           label=label, alpha=0.6, s=50)
            ax16.set_xlabel('Prop Kuning')
            ax16.set_ylabel('Mean V')
            ax16.set_title('Feature Interaction: Kuning vs Brightness')
            ax16.legend()
    
    def _print_statistical_summaries(self):
        """Print comprehensive statistical summaries"""
        print("\n" + "="*60)
        print("📈 STATISTICAL SUMMARIES")
        print("="*60)
        
        # Basic statistics
        print("\n1. BASIC DATASET STATISTICS:")
        print("-" * 40)
        print(f"Total samples: {len(self.df)}")
        print(f"Number of features: {len(self.feature_cols)}")
        print(f"Classes: {list(self.df['Label'].unique())}")
        
        class_counts = self.df['Label'].value_counts()
        for label, count in class_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {label}: {count} samples ({percentage:.1f}%)")
        
        # Feature statistics by class
        print("\n2. FEATURE STATISTICS BY CLASS:")
        print("-" * 40)
        for label in self.df['Label'].unique():
            print(f"\n{label.upper()} CLASS:")
            class_data = self.df[self.df['Label'] == label][self.feature_cols]
            print(class_data.describe().round(3))
        
        # Statistical tests
        print("\n3. STATISTICAL SIGNIFICANCE TESTS:")
        print("-" * 40)
        
        X = self.df[self.feature_cols]
        y = self.df['Label']
        
        # ANOVA F-test
        f_stats, p_values = f_classif(X, y)
        
        print("ANOVA F-test results (features ranked by importance):")
        importance_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'F_Score': f_stats,
            'P_Value': p_values,
            'Significant': p_values < 0.05
        }).sort_values('F_Score', ascending=False)
        
        for _, row in importance_df.head(8).iterrows():
            significance = "***" if row['P_Value'] < 0.001 else "**" if row['P_Value'] < 0.01 else "*" if row['P_Value'] < 0.05 else ""
            print(f"  {row['Feature']:<15}: F={row['F_Score']:.3f}, p={row['P_Value']:.4f} {significance}")
        
        # Correlation insights
        print("\n4. CORRELATION INSIGHTS:")
        print("-" * 40)
        
        corr_matrix = X.corr()
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.5:  # High correlation threshold
                    high_corr_pairs.append((
                        corr_matrix.columns[i],
                        corr_matrix.columns[j],
                        corr_val
                    ))
        
        if high_corr_pairs:
            print("High correlations (|r| > 0.5):")
            for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
                print(f"  {feat1} <-> {feat2}: r={corr:.3f}")
        else:
            print("No high correlations found (|r| > 0.5)")

# =============================================================================
# 6. DATA PREPROCESSING
# =============================================================================

class DataPreprocessor:
    """
    Advanced data preprocessing untuk machine learning
    """
    
    def __init__(self):
        self.scaler = None
        self.feature_selector = None
        self.label_encoder = None
        self.selected_features = None
        
    def preprocess_data(self, df, test_size=0.2, feature_selection_k=10):
        """
        Comprehensive data preprocessing pipeline
        """
        print("\n" + "="*60)
        print("🔧 DATA PREPROCESSING PIPELINE")
        print("="*60)
        
        # Separate features and labels
        feature_cols = [col for col in df.columns if col not in ['Label', 'Filename']]
        X = df[feature_cols].values
        y = df['Label'].values
        
        print(f"Original features shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Handle any remaining NaN or inf values
        if np.isnan(X).any() or np.isinf(X).any():
            print("⚠️  Fixing NaN/inf values...")
            X = np.nan_to_num(X, nan=0.0, posinf=999999.0, neginf=-999999.0)
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        label_mapping = dict(zip(self.label_encoder.classes_, 
                                range(len(self.label_encoder.classes_))))
        print(f"Label encoding: {label_mapping}")
        
        # Train-test split with stratification
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
            )
            print(f"Stratified split successful")
        except ValueError as e:
            print(f"⚠️  Stratification failed ({e}), using random split...")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y_encoded, test_size=test_size, random_state=42
            )
        
        # Feature scaling using RobustScaler (better for outliers than StandardScaler)
        print("Applying RobustScaler...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Feature selection
        print(f"Selecting top {feature_selection_k} features...")
        self.feature_selector = SelectKBest(f_classif, k=min(feature_selection_k, len(feature_cols)))
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = self.feature_selector.transform(X_test_scaled)
        
        # Get selected feature names
        selected_indices = self.feature_selector.get_support(indices=True)
        self.selected_features = [feature_cols[i] for i in selected_indices]
        
        print(f"Selected features: {self.selected_features}")
        print(f"Final feature shape: {X_train_selected.shape}")
        
        return (X_train_selected, X_test_selected, y_train, y_test, 
                self.selected_features, feature_cols)

# =============================================================================
# 7. MACHINE LEARNING MODELS (IMAGE-OPTIMIZED)
# =============================================================================

class ImageOptimizedMLPipeline:
    """
    Pipeline ML yang dioptimasi untuk data image-derived features
    """
    
    def __init__(self):
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        
    def initialize_models(self):
        """
        Initialize models yang cocok untuk image-derived features
        """
        # Models yang terbukti bagus untuk image features
        self.models = {
            # Ensemble methods - excellent for image features
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            
            'Extra Trees': ExtraTreesClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42,
                n_jobs=-1
            ),
            
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=150,
                learning_rate=0.1,
                max_depth=6,
                min_samples_split=4,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=42
            ),
            
            # SVM - good for high-dimensional image features
            'SVM (RBF)': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                random_state=42
            ),
            
            # k-NN - works well with image similarity
            'k-NN': KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='minkowski',
                p=2
            ),
            
            # Logistic Regression - good baseline
            'Logistic Regression': LogisticRegression(
                C=1.0,
                max_iter=2000,
                solver='liblinear',
                random_state=42
            ),
            
            # Naive Bayes - simple but effective
            'Naive Bayes': GaussianNB()
        }
        
        print(f"Initialized {len(self.models)} models optimized for image features")
    
    def train_and_evaluate_models(self, X_train, X_test, y_train, y_test):
        """
        Train and evaluate all models with cross-validation
        """
        print("\n" + "="*60)
        print("🤖 TRAINING IMAGE-OPTIMIZED ML MODELS")
        print("="*60)
        
        if not self.models:
            self.initialize_models()
        
        # Cross-validation setup
        cv_folds = min(5, len(y_train)//10) if len(y_train) > 20 else 3
        cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            try:
                # Cross-validation
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=cv, 
                    scoring='accuracy', n_jobs=-1
                )
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Test predictions
                y_pred = model.predict(X_test)
                y_pred_proba = None
                
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)
                
                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
                f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
                
                # AUC for binary classification
                auc_score = None
                if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
                    auc_score = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                # Store results
                self.results[name] = {
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'test_accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1,
                    'auc_score': auc_score if auc_score else 0.0,
                    'cv_scores': cv_scores
                }
                
                print(f"  ✅ CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std()*2:.4f})")
                print(f"     Test Accuracy: {accuracy:.4f}")
                print(f"     F1 Score: {f1:.4f}")
                if auc_score:
                    print(f"     AUC Score: {auc_score:.4f}")
                
            except Exception as e:
                print(f"  ❌ Error training {name}: {e}")
                continue
        
        # Train CNN if TensorFlow is available
        if TENSORFLOW_AVAILABLE:
            try:
                self.train_cnn_features_model(X_train, X_test, y_train, y_test)
            except Exception as e:
                print(f"  ❌ Error training CNN Features model: {e}")
    
    def train_cnn_features_model(self, X_train, X_test, y_train, y_test):
        """
        Train a neural network optimized for image-derived features
        """
        print(f"\nTraining CNN-Features Neural Network...")
        
        # Build neural network architecture optimized for image features
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers - designed for image feature patterns
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            # Output layer
            Dense(1, activation='sigmoid')
        ])
        
        # Compile with optimized settings for image features
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=15, 
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=0
        )
        
        # Train
        history = model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=min(32, len(X_train)//4),
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Evaluate
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = (y_pred_proba > 0.5).astype(int).flatten()
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        auc_score = roc_auc_score(y_test, y_pred_proba.flatten())
        
        # Store results
        self.models['CNN-Features NN'] = model
        self.results['CNN-Features NN'] = {
            'cv_mean': accuracy,  # Use test accuracy as proxy
            'cv_std': 0.0,
            'test_accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score,
            'training_history': history.history
        }
        
        print(f"  ✅ CNN-Features NN Test Accuracy: {accuracy:.4f}")
        print(f"     F1 Score: {f1:.4f}")
        print(f"     AUC Score: {auc_score:.4f}")
    
    def display_comprehensive_results(self):
        """
        Display comprehensive model comparison results
        """
        print("\n" + "="*80)
        print("🏆 COMPREHENSIVE MODEL COMPARISON RESULTS")
        print("="*80)
        
        if not self.results:
            print("❌ No models were successfully trained!")
            return None, None
        
        # Create results DataFrame
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('test_accuracy', ascending=False)
        
        # Display results table
        print("\nModel Performance Summary:")
        print("-" * 80)
        print(f"{'Model':<20} {'CV Acc':<8} {'CV Std':<8} {'Test Acc':<9} {'Precision':<9} {'Recall':<8} {'F1':<8} {'AUC':<8}")
        print("-" * 80)
        
        for model_name, row in results_df.iterrows():
            print(f"{model_name:<20} {row['cv_mean']:.4f}   {row['cv_std']:.4f}   "
                  f"{row['test_accuracy']:.4f}    {row['precision']:.4f}    "
                  f"{row['recall']:.4f}   {row['f1_score']:.4f}   {row['auc_score']:.4f}")
        
        # Select best model
        self.best_model_name = results_df.index[0]
        self.best_model = self.models[self.best_model_name]
        
        print(f"\n🏆 BEST MODEL: {self.best_model_name}")
        print(f"   Test Accuracy: {results_df.loc[self.best_model_name, 'test_accuracy']:.4f}")
        print(f"   F1 Score: {results_df.loc[self.best_model_name, 'f1_score']:.4f}")
        print(f"   AUC Score: {results_df.loc[self.best_model_name, 'auc_score']:.4f}")
        
        # Visualize results
        self.plot_model_comparison(results_df)
        
        return self.best_model_name, self.best_model
    
    def plot_model_comparison(self, results_df):
        """
        Plot comprehensive model comparison
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Test Accuracy Comparison
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(results_df)), results_df['test_accuracy'], 
                      color='skyblue', alpha=0.8)
        ax1.set_xticks(range(len(results_df)))
        ax1.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax1.set_ylabel('Test Accuracy')
        ax1.set_title('Test Accuracy Comparison')
        ax1.set_ylim(0, 1)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 2. F1 Score Comparison
        ax2 = axes[0, 1]
        bars = ax2.bar(range(len(results_df)), results_df['f1_score'], 
                      color='lightcoral', alpha=0.8)
        ax2.set_xticks(range(len(results_df)))
        ax2.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('F1 Score Comparison')
        ax2.set_ylim(0, 1)
        
        # Add value labels
        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        
        # 3. Cross-validation Performance
        ax3 = axes[1, 0]
        ax3.errorbar(range(len(results_df)), results_df['cv_mean'], 
                    yerr=results_df['cv_std'], fmt='o', capsize=5)
        ax3.set_xticks(range(len(results_df)))
        ax3.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax3.set_ylabel('CV Accuracy')
        ax3.set_title('Cross-Validation Performance')
        ax3.grid(True, alpha=0.3)
        
        # 4. Multi-metric Radar Chart (for top 5 models)
        ax4 = axes[1, 1]
        top_5_models = results_df.head(5)
        
        metrics = ['test_accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC']
        
        for i, (model_name, row) in enumerate(top_5_models.iterrows()):
            values = [row[metric] for metric in metrics]
            values += values[:1]  # Complete the circle
            
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            ax4.plot(angles, values, 'o-', linewidth=2, label=model_name[:10])
            ax4.fill(angles, values, alpha=0.1)
        
        ax4.set_xticks(angles[:-1])
        ax4.set_xticklabels(metric_labels)
        ax4.set_ylim(0, 1)
        ax4.set_title('Multi-Metric Performance (Top 5)')
        ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def hyperparameter_tuning(self, X_train, y_train):
        """
        Hyperparameter tuning untuk best model
        """
        if self.best_model_name is None:
            print("❌ No best model found for hyperparameter tuning")
            return self.best_model
        
        print(f"\n🔧 HYPERPARAMETER TUNING FOR {self.best_model_name}")
        print("="*60)
        
        # Define parameter grids for different models
        param_grids = {
            'Random Forest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2']
            },
            'Extra Trees': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 15, 20, None],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 3],
                'max_features': ['sqrt', 'log2']
            },
            'Gradient Boosting': {
                'n_estimators': [100, 150, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [4, 6, 8],
                'min_samples_split': [2, 4, 6]
            },
            'SVM (RBF)': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.001, 0.01, 0.1]
            },
            'k-NN': {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan', 'minkowski']
            },
            'Logistic Regression': {
                'C': [0.1, 1, 10, 100],
                'solver': ['liblinear', 'lbfgs'],
                'max_iter': [1000, 2000]
            }
        }
        
        if self.best_model_name in param_grids:
            param_grid = param_grids[self.best_model_name]
            
            # Create base model
            base_models = {
                'Random Forest': RandomForestClassifier(random_state=42, n_jobs=-1),
                'Extra Trees': ExtraTreesClassifier(random_state=42, n_jobs=-1),
                'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
                'k-NN': KNeighborsClassifier(),
                'Logistic Regression': LogisticRegression(random_state=42)
            }
            
            base_model = base_models[self.best_model_name]
            
            # Grid search with cross-validation
            cv_folds = min(3, len(y_train)//20) if len(y_train) > 60 else 2
            
            grid_search = GridSearchCV(
                base_model, param_grid,
                cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
                scoring='accuracy',
                n_jobs=-1,
                verbose=1
            )
            
            print("Starting grid search...")
            grid_search.fit(X_train, y_train)
            
            print(f"✅ Best parameters: {grid_search.best_params_}")
            print(f"✅ Best CV score: {grid_search.best_score_:.4f}")
            
            self.best_model = grid_search.best_estimator_
            return self.best_model
        
        else:
            print(f"Hyperparameter tuning not implemented for {self.best_model_name}")
            return self.best_model

# =============================================================================
# 8. REAL-TIME WEBCAM CLASSIFIER
# =============================================================================

class RealTimeWebcamClassifier:
    """
    Sistem klasifikasi real-time menggunakan webcam
    """
    
    def __init__(self, model, scaler, feature_extractor, label_encoder, selected_features):
        self.model = model
        self.scaler = scaler
        self.feature_extractor = feature_extractor
        self.label_encoder = label_encoder
        self.selected_features = selected_features
        self.prediction_history = []
        
    def predict_from_roi(self, roi):
        """Prediksi dari ROI webcam"""
        try:
            # Extract all features
            all_features = self.feature_extractor.extract_features(roi)
            
            # Select only the features used in training
            if hasattr(self.feature_extractor, 'feature_names'):
                feature_dict = dict(zip(self.feature_extractor.feature_names, all_features))
                selected_feature_values = np.array([feature_dict.get(name, 0) for name in self.selected_features])
            else:
                # Fallback: use first N features
                selected_feature_values = all_features[:len(self.selected_features)]
            
            # Scale features
            features_scaled = self.scaler.transform(selected_feature_values.reshape(1, -1))
            
            # Make prediction
            if hasattr(self.model, 'predict_proba'):
                prediction_encoded = self.model.predict(features_scaled)[0]
                confidence = np.max(self.model.predict_proba(features_scaled))
            else:
                # Handle TensorFlow model
                prediction_proba = self.model.predict(features_scaled, verbose=0)[0][0]
                prediction_encoded = int(prediction_proba > 0.5)
                confidence = prediction_proba if prediction_proba > 0.5 else (1 - prediction_proba)
            
            # Decode prediction
            prediction = self.label_encoder.inverse_transform([prediction_encoded])[0]
            
            return prediction, confidence, selected_feature_values
            
        except Exception as e:
            print(f"Prediction error: {e}")
            return "Error", 0.0, np.zeros(len(self.selected_features))
    
    def smooth_predictions(self, prediction, confidence):
        """Smooth predictions menggunakan history"""
        self.prediction_history.append((prediction, confidence))
        
        # Keep only last 10 predictions
        if len(self.prediction_history) > 10:
            self.prediction_history.pop(0)
        
        # Get most common prediction in recent history
        recent_predictions = [p[0] for p in self.prediction_history[-5:]]
        recent_confidences = [p[1] for p in self.prediction_history[-5:]]
        
        # Most frequent prediction
        smooth_prediction = max(set(recent_predictions), key=recent_predictions.count)
        avg_confidence = np.mean(recent_confidences)
        
        return smooth_prediction, avg_confidence
    
    def run_webcam_classification(self):
        """
        Real-time fruit classification menggunakan webcam
        """
        print("\n" + "="*60)
        print("🎥 REAL-TIME FRUIT CLASSIFICATION WEBCAM")
        print("="*60)
        print("Controls:")
        print("  'q' or ESC : Quit")
        print("  's'        : Save current prediction")
        print("  'r'        : Reset prediction history")
        print("  'h'        : Toggle feature display")
        print("  'f'        : Toggle fullscreen")
        print("  SPACE      : Pause/Resume")
        print("="*60)
        
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Error: Cannot open webcam!")
            return
        
        # Webcam settings
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # State variables
        save_counter = 1
        show_features = True
        fullscreen = False
        paused = False
        
        # Create output directories
        os.makedirs('webcam_predictions', exist_ok=True)
        
        print("🎯 Webcam started! Position fruit in the green rectangle...")
        
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading from webcam")
                    break
            
            # Define ROI
            h, w, _ = frame.shape
            roi_size = min(h, w) // 3
            x1 = (w - roi_size) // 2
            y1 = (h - roi_size) // 2
            x2 = x1 + roi_size
            y2 = y1 + roi_size
            
            roi = frame[y1:y2, x1:x2]
            
            # Make prediction
            prediction, confidence, features = self.predict_from_roi(roi)
            
            # Smooth predictions
            if prediction != "Error":
                smooth_pred, smooth_conf = self.smooth_predictions(prediction, confidence)
            else:
                smooth_pred, smooth_conf = "Error", 0.0
            
            # Create overlay
            overlay = frame.copy()
            
            # ROI rectangle color
            if smooth_pred == 'matang':
                roi_color = (0, 255, 0)  # Green
                bg_color = (0, 50, 0)
            elif smooth_pred == 'mentah':
                roi_color = (0, 165, 255)  # Orange
                bg_color = (0, 30, 50)
            else:
                roi_color = (0, 0, 255)  # Red
                bg_color = (0, 0, 50)
            
            # Draw ROI
            cv2.rectangle(overlay, (x1, y1), (x2, y2), roi_color, 4)
            
            # Main prediction display
            pred_text = f"PREDIKSI: {smooth_pred.upper()}"
            conf_text = f"CONFIDENCE: {smooth_conf:.1%}"
            
            # Background for text
            text_bg_y1 = y1 - 80
            text_bg_y2 = y1 - 10
            cv2.rectangle(overlay, (x1, text_bg_y1), (x2 + 100, text_bg_y2), bg_color, -1)
            cv2.rectangle(overlay, (x1, text_bg_y1), (x2 + 100, text_bg_y2), roi_color, 2)
            
            # Draw text
            cv2.putText(overlay, pred_text, (x1 + 10, y1 - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(overlay, conf_text, (x1 + 10, y1 - 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Feature display
            if show_features and prediction != "Error":
                self.display_features(overlay, features)
            
            # Status bar
            self.display_status_bar(overlay, w, h)
            
            # History indicator
            history_text = f"History: {len(self.prediction_history)}/10"
            cv2.putText(overlay, history_text, (w - 200, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Pause indicator
            if paused:
                cv2.putText(overlay, "PAUSED", (w//2 - 80, h//2), 
                           cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255), 3)
            
            # Display
            window_name = "Fruit Ripeness Classifier"
            if fullscreen:
                cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            
            cv2.imshow(window_name, overlay)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q') or key == 27:  # Quit
                break
            elif key == ord('s') and prediction != "Error":  # Save
                self.save_prediction(roi, overlay, prediction, confidence, smooth_pred, smooth_conf, features, save_counter)
                save_counter += 1
            elif key == ord('r'):  # Reset
                self.prediction_history = []
                print("🔄 Prediction history reset")
            elif key == ord('h'):  # Toggle features
                show_features = not show_features
                print(f"📊 Feature display: {'ON' if show_features else 'OFF'}")
            elif key == ord('f'):  # Toggle fullscreen
                fullscreen = not fullscreen
                print(f"🖥️  Fullscreen: {'ON' if fullscreen else 'OFF'}")
            elif key == ord(' '):  # Pause
                paused = not paused
                print(f"⏸️  {'Paused' if paused else 'Resumed'}")
        
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n✅ Classification session ended. Saved {save_counter - 1} predictions.")
    
    def display_features(self, overlay, features):
        """Display feature values on overlay"""
        if len(features) < 6:
            return
        
        feature_display = [
            ("Selected Features:", ""),
        ]
        
        # Display selected features
        for i, feature_name in enumerate(self.selected_features[:6]):  # Show first 6
            if i < len(features):
                if 'Prop' in feature_name:
                    value_str = f"{features[i]*100:.1f}%"
                elif 'Mean H' in feature_name:
                    value_str = f"{features[i]:.0f}°"
                else:
                    value_str = f"{features[i]:.1f}"
                feature_display.append((f"{feature_name[:10]}", value_str))
        
        # Features background
        cv2.rectangle(overlay, (10, 60), (300, 60 + len(feature_display)*25), (0, 0, 0), -1)
        cv2.rectangle(overlay, (10, 60), (300, 60 + len(feature_display)*25), (100, 100, 100), 2)
        
        for i, (name, value) in enumerate(feature_display):
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            text = f"{name}: {value}" if value else name
            cv2.putText(overlay, text, (20, 85 + i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def display_status_bar(self, overlay, w, h):
        """Display status bar at bottom"""
        status_y = h - 60
        cv2.rectangle(overlay, (0, status_y), (w, h), (0, 0, 0), -1)
        
        controls = "Controls: 'q'=quit | 's'=save | 'r'=reset | 'h'=features | 'f'=fullscreen | SPACE=pause"
        cv2.putText(overlay, controls, (10, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        cv2.putText(overlay, f"Time: {timestamp}", (w - 150, status_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def save_prediction(self, roi, overlay, prediction, confidence, smooth_pred, smooth_conf, features, counter):
        """Save prediction dengan detail lengkap"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save ROI
        roi_path = f"webcam_predictions/roi_{counter:03d}_{timestamp}.jpg"
        cv2.imwrite(roi_path, roi)
        
        # Save full frame
        frame_path = f"webcam_predictions/frame_{counter:03d}_{timestamp}.jpg"
        cv2.imwrite(frame_path, overlay)
        
        # Save detailed info
        info_path = f"webcam_predictions/info_{counter:03d}_{timestamp}.txt"
        with open(info_path, 'w') as f:
            f.write("FRUIT RIPENESS PREDICTION REPORT\n")
            f.write("="*40 + "\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write(f"Counter: {counter:03d}\n\n")
            
            f.write("PREDICTIONS:\n")
            f.write(f"Raw Prediction: {prediction}\n")
            f.write(f"Raw Confidence: {confidence:.4f}\n")
            f.write(f"Smooth Prediction: {smooth_pred}\n")
            f.write(f"Smooth Confidence: {smooth_conf:.4f}\n")
            f.write(f"History Length: {len(self.prediction_history)}\n\n")
            
            f.write("SELECTED FEATURES:\n")
            for i, name in enumerate(self.selected_features):
                if i < len(features):
                    f.write(f"{name:>15}: {features[i]:8.3f}\n")
            
            f.write(f"\nFILES SAVED:\n")
            f.write(f"ROI Image: {roi_path}\n")
            f.write(f"Full Frame: {frame_path}\n")
        
        print(f"💾 Saved prediction {counter:03d}: {smooth_pred} (confidence: {smooth_conf:.1%})")

# =============================================================================
# 9. MAIN PIPELINE EXECUTION
# =============================================================================

def main():
    """
    MAIN PIPELINE EXECUTION
    
    STRUKTUR PENGEMBANGAN:
    1. Data Loading & Comprehensive EDA
    2. Advanced Data Preprocessing
    3. Image-Optimized Model Training & Comparison
    4. Hyperparameter Tuning
    5. Real-time Webcam Deployment
    """
    
    print("🍎" * 20)
    print("FRUIT RIPENESS CLASSIFICATION SYSTEM - IMPROVED")
    print("🍎" * 20)
    
    # Configuration
    DATASET_URL = "https://docs.google.com/spreadsheets/d/1XrjguhS1MDdgQbBTfzKS35WUcVxPT4tJtqGk9mh4KTQ/edit?usp=sharing"
    
    # =============================================================================
    # STEP 1: DATASET LOADING
    # =============================================================================
    print("\n📂 STEP 1: LOADING DATASET")
    print("-" * 40)
    
    dataset_manager = FruitDatasetManager(DATASET_URL)
    if not dataset_manager.load_dataset():
        print("❌ Failed to load dataset")
        return
    
    # Create sample images for testing
    dataset_manager.download_sample_images()
    df = dataset_manager.df
    
    # =============================================================================
    # STEP 2: COMPREHENSIVE EXPLORATORY DATA ANALYSIS
    # =============================================================================
    print("\n📊 STEP 2: COMPREHENSIVE EXPLORATORY DATA ANALYSIS")
    print("-" * 40)
    
    eda_analyzer = EDAnalyzer(df)
    eda_analyzer.comprehensive_eda()
    
    # =============================================================================
    # STEP 3: ADVANCED DATA PREPROCESSING
    # =============================================================================
    print("\n🔧 STEP 3: ADVANCED DATA PREPROCESSING")
    print("-" * 40)
    
    preprocessor = DataPreprocessor()
    (X_train, X_test, y_train, y_test, 
     selected_features, all_features) = preprocessor.preprocess_data(df, feature_selection_k=8)
    
    # =============================================================================
    # STEP 4: IMAGE-OPTIMIZED MODEL TRAINING
    # =============================================================================
    print("\n🤖 STEP 4: IMAGE-OPTIMIZED MODEL TRAINING")
    print("-" * 40)
    
    ml_pipeline = ImageOptimizedMLPipeline()
    ml_pipeline.train_and_evaluate_models(X_train, X_test, y_train, y_test)
    
    # =============================================================================
    # STEP 5: MODEL SELECTION & HYPERPARAMETER TUNING
    # =============================================================================
    print("\n🏆 STEP 5: MODEL SELECTION & HYPERPARAMETER TUNING")
    print("-" * 40)
    
    best_model_name, best_model = ml_pipeline.display_comprehensive_results()
    
    if best_model_name is None:
        print("❌ No models were successfully trained. Exiting...")
        return
    
    # Hyperparameter tuning
    tuned_model = ml_pipeline.hyperparameter_tuning(X_train, y_train)
    
    # =============================================================================
    # STEP 6: MODEL SAVING
    # =============================================================================
    print("\n💾 STEP 6: SAVING MODEL")
    print("-" * 40)
    
    model_data = {
        'model': tuned_model,
        'scaler': preprocessor.scaler,
        'feature_selector': preprocessor.feature_selector,
        'label_encoder': preprocessor.label_encoder,
        'feature_extractor': FruitFeatureExtractor(),
        'selected_features': selected_features,
        'all_features': all_features,
        'model_name': best_model_name,
        'training_timestamp': datetime.now().isoformat(),
        'model_results': ml_pipeline.results[best_model_name]
    }
    
    joblib.dump(model_data, 'fruit_ripeness_model_improved.pkl')
    print(f"✅ Model saved: fruit_ripeness_model_improved.pkl")
    print(f"   Model Type: {best_model_name}")
    print(f"   Test Accuracy: {ml_pipeline.results[best_model_name]['test_accuracy']:.1%}")
    print(f"   F1 Score: {ml_pipeline.results[best_model_name]['f1_score']:.1%}")
    
    # =============================================================================
    # STEP 7: REAL-TIME WEBCAM CLASSIFICATION
    # =============================================================================
    print("\n🎥 STEP 7: REAL-TIME WEBCAM CLASSIFICATION")
    print("-" * 40)
    print("Starting webcam classification...")
    
    webcam_classifier = RealTimeWebcamClassifier(
        tuned_model, 
        preprocessor.scaler, 
        FruitFeatureExtractor(), 
        preprocessor.label_encoder,
        selected_features
    )
    
    try:
        webcam_classifier.run_webcam_classification()
    except KeyboardInterrupt:
        print("\n⏹️  Classification interrupted by user")
    except Exception as e:
        print(f"\n❌ Error in webcam classification: {e}")
    
    print("\n🎉 SYSTEM COMPLETED SUCCESSFULLY!")
    print("Files generated:")
    print("  - fruit_ripeness_model_improved.pkl (trained model)")
    print("  - comprehensive_eda_analysis.png (exploratory analysis)")
    print("  - model_comparison_results.png (model comparison)")
    print("  - webcam_predictions/ (webcam results)")
    print("  - dataset_images/ (sample images)")

if __name__ == "__main__":
    main()