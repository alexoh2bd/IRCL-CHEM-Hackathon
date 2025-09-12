import pandas as pd
import numpy as np
import json
import ast
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

class EnhancedResponseAnalyzer:
    def __init__(self, csv_path: str):
        """
        Initialize the Enhanced ResponseAnalyzer with comprehensive positive/negative metrics
        
        Args:
            csv_path (str): Path to the processedchem.csv file
        """
        self.csv_path = csv_path
        self.df = None
        self.response_confidence_data = []
        
    def load_data(self):
        """Load the CSV data with specified column headers"""
        try:
            # Load CSV with specified column headers
            self.df = pd.read_csv(self.csv_path)
            
            # Assign column headers if they don't match
            expected_columns = ['ID', 'Query', 'OgPositive', 'Instruction', 'Response']
            if list(self.df.columns) != expected_columns:
                self.df.columns = expected_columns
                
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            print(f"Columns: {list(self.df.columns)}")
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def analyze_confidence_distribution(self, positive_threshold: float = 0.7, negative_threshold: float = 0.3):
        """Analyze the distribution of response confidence levels with comprehensive positive/negative metrics"""
        if self.df is None:
            print("Data not loaded. Please run load_data() first.")
            return None, None
            
        print("Analyzing confidence score distribution...")
        all_confidence_scores = []
        response_counts = []
        positive_scores = []
        negative_scores = []
        neutral_scores = []
        
        # Clear previous data
        self.response_confidence_data = []
        
        # Process all rows in the dataset
        for idx, row in self.df.iterrows():
            try:
                responses = ast.literal_eval(row['Response'])
                if not isinstance(responses, list):
                    responses = [responses]
                    
                scores = [float(r['confidence']) for r in responses] 
                # labels = [r['label'] for r in responses]                
                all_confidence_scores.extend(scores)
                response_counts.append(len(responses))
                
                # Categorize scores by thresholds
                for i in range(len(responses)):
                    if responses[i]['label'] == "positive":
                        positive_scores.append(scores[i])
                    else:
                        negative_scores.append(scores[i])

                
                # Store detailed data for each row
                self.response_confidence_data.append({
                    'ID': row['ID'],
                    'Query': row['Query'],
                    'num_responses': len(responses),
                    'confidence_scores': scores,
                    'responses': responses
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Convert to numpy arrays for analysis
        confidence_array = np.array(all_confidence_scores)
        positive_array = np.array(positive_scores)
        negative_array = np.array(negative_scores)
        neutral_array = np.array(neutral_scores)
        
        if len(confidence_array) == 0:
            print("No confidence scores found in the data.")
            return None, None
            
        # Calculate overall statistics
        stats = {
            'total_responses': len(all_confidence_scores),
            'mean_confidence': np.mean(confidence_array),
            'median_confidence': np.median(confidence_array),
            'std_confidence': np.std(confidence_array),
            'min_confidence': np.min(confidence_array),
            'max_confidence': np.max(confidence_array),
            'q25': np.percentile(confidence_array, 25),
            'q75': np.percentile(confidence_array, 75)
        }
        
        # Calculate positive label metrics
        positive_stats = {}
        if len(positive_array) > 0:
            positive_stats = {
                'count': len(positive_array),
                'percentage': len(positive_array) / len(confidence_array) * 100,
                'mean': np.mean(positive_array),
                'median': np.median(positive_array),
                'std': np.std(positive_array),
                'min': np.min(positive_array),
                'max': np.max(positive_array),
                'q25': np.percentile(positive_array, 25),
                'q75': np.percentile(positive_array, 75),
                'variance': np.var(positive_array),
                'skewness': self._calculate_skewness(positive_array),
                'kurtosis': self._calculate_kurtosis(positive_array)
            }
        
        # Calculate negative label metrics
        negative_stats = {}
        if len(negative_array) > 0:
            negative_stats = {
                'count': len(negative_array),
                'percentage': len(negative_array) / len(confidence_array) * 100,
                'mean': np.mean(negative_array),
                'median': np.median(negative_array),
                'std': np.std(negative_array),
                'min': np.min(negative_array),
                'max': np.max(negative_array),
                'q25': np.percentile(negative_array, 25),
                'q75': np.percentile(negative_array, 75),
                'variance': np.var(negative_array),
                'skewness': self._calculate_skewness(negative_array),
                'kurtosis': self._calculate_kurtosis(negative_array)
            }
        
        # Calculate neutral label metrics
        neutral_stats = {}
        if len(neutral_array) > 0:
            neutral_stats = {
                'count': len(neutral_array),
                'percentage': len(neutral_array) / len(confidence_array) * 100,
                'mean': np.mean(neutral_array),
                'median': np.median(neutral_array),
                'std': np.std(neutral_array),
                'min': np.min(neutral_array),
                'max': np.max(neutral_array),
                'q25': np.percentile(neutral_array, 25),
                'q75': np.percentile(neutral_array, 75),
                'variance': np.var(neutral_array),
                'skewness': self._calculate_skewness(neutral_array),
                'kurtosis': self._calculate_kurtosis(neutral_array)
            }
        
        # Print comprehensive analysis
        self._print_analysis_results(stats, positive_stats, negative_stats, neutral_stats, 
                                   positive_threshold, negative_threshold, response_counts)
        
        # Return comprehensive stats
        all_stats = {
            'overall': stats,
            'positive': positive_stats,
            'negative': negative_stats,
            'neutral': neutral_stats,
            'thresholds': {'positive': positive_threshold, 'negative': negative_threshold}
        }
        
        return all_stats, confidence_array
    
    def _calculate_skewness(self, data):
        """Calculate skewness of the data"""
        if len(data) < 3:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of the data"""
        if len(data) < 4:
            return 0
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _print_analysis_results(self, stats, positive_stats, negative_stats, neutral_stats, 
                              positive_threshold, negative_threshold, response_counts):
        """Print comprehensive analysis results"""
        
        print("\n" + "="*60)
        print("COMPREHENSIVE CONFIDENCE SCORE ANALYSIS")
        print("="*60)
        
        # Overall statistics
        print(f"\n=== OVERALL DISTRIBUTION ===")
        print(f"Total responses analyzed: {stats['total_responses']}")
        print(f"Mean confidence: {stats['mean_confidence']:.4f}")
        print(f"Median confidence: {stats['median_confidence']:.4f}")
        print(f"Standard deviation: {stats['std_confidence']:.4f}")
        print(f"Range: {stats['min_confidence']:.4f} - {stats['max_confidence']:.4f}")
        print(f"Interquartile range: {stats['q25']:.4f} - {stats['q75']:.4f}")
        
        # Positive label metrics
        print(f"\n=== POSITIVE LABELS  ===")
        if positive_stats:
            print(f"Count: {positive_stats['count']} ({positive_stats['percentage']:.1f}% of total)")
            print(f"Mean: {positive_stats['mean']:.4f}")
            print(f"Median: {positive_stats['median']:.4f}")
            print(f"Std Dev: {positive_stats['std']:.4f}")
            print(f"Variance: {positive_stats['variance']:.4f}")
            print(f"Range: {positive_stats['min']:.4f} - {positive_stats['max']:.4f}")
            print(f"IQR: {positive_stats['q25']:.4f} - {positive_stats['q75']:.4f}")
            print(f"Skewness: {positive_stats['skewness']:.4f}")
            print(f"Kurtosis: {positive_stats['kurtosis']:.4f}")
        else:
            print("No positive responses found above threshold")
        
        # Negative label metrics
        print(f"\n=== NEGATIVE LABELS ===")
        if negative_stats:
            print(f"Count: {negative_stats['count']} ({negative_stats['percentage']:.1f}% of total)")
            print(f"Mean: {negative_stats['mean']:.4f}")
            print(f"Median: {negative_stats['median']:.4f}")
            print(f"Std Dev: {negative_stats['std']:.4f}")
            print(f"Variance: {negative_stats['variance']:.4f}")
            print(f"Range: {negative_stats['min']:.4f} - {negative_stats['max']:.4f}")
            print(f"IQR: {negative_stats['q25']:.4f} - {negative_stats['q75']:.4f}")
            print(f"Skewness: {negative_stats['skewness']:.4f}")
            print(f"Kurtosis: {negative_stats['kurtosis']:.4f}")
        else:
            print("No negative responses found below threshold")
        
        
        # Response count statistics
        response_count_array = np.array(response_counts)
        print(f"\n=== RESPONSE COUNT STATISTICS ===")
        print(f"Total queries processed: {len(response_counts)}")
        print(f"Average responses per query: {np.mean(response_count_array):.2f}")
        print(f"Max responses per query: {np.max(response_count_array)}")
        print(f"Min responses per query: {np.min(response_count_array)}")
        
        # Calculate label distribution per query
        queries_with_positives = sum(1 for data in self.response_confidence_data 
                                   if any(score >= positive_threshold for score in data['confidence_scores']))
        queries_with_negatives = sum(1 for data in self.response_confidence_data 
                                   if any(score <= negative_threshold for score in data['confidence_scores']))
        queries_with_both = sum(1 for data in self.response_confidence_data 
                              if any(score >= positive_threshold for score in data['confidence_scores']) and
                                 any(score <= negative_threshold for score in data['confidence_scores']))
        
        print(f"\n=== QUERY-LEVEL LABEL DISTRIBUTION ===")
        print(f"Queries with at least one positive response: {queries_with_positives} ({queries_with_positives/len(response_counts)*100:.1f}%)")
        print(f"Queries with at least one negative response: {queries_with_negatives} ({queries_with_negatives/len(response_counts)*100:.1f}%)")
        print(f"Queries with both positive and negative responses: {queries_with_both} ({queries_with_both/len(response_counts)*100:.1f}%)")
    
    def filter_responses(self, positive_threshold: float = 0.7, negative_threshold: float = 0.3) -> List[Dict]:
        """
        Filter responses based on confidence thresholds
        
        Args:
            positive_threshold (float): Minimum confidence for positive responses
            negative_threshold (float): Maximum confidence for negative responses
            
        Returns:
            List[Dict]: Filtered results for each query
        """
        filtered_results = []
        
        for data in self.response_confidence_data:
            result = {
                'ID': data['ID'],
                'Query': data['Query'],
                'top_positive': None,
                'bottom_negatives': [],
                'all_responses': data['responses']
            }
            
            # Create list of (response, confidence) tuples
            response_confidence_pairs = []
            for i, response in enumerate(data['responses']):
                if i < len(data['confidence_scores']):
                    confidence = data['confidence_scores'][i]
                    response_confidence_pairs.append((response, confidence))
            
            # Sort by confidence score (descending)
            response_confidence_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Find top positive response (above threshold)
            for response, confidence in response_confidence_pairs:
                if confidence >= positive_threshold:
                    result['top_positive'] = {
                        'response': response,
                        'confidence': confidence
                    }
                    break
            
            # Find bottom 3 negative responses (below threshold)
            negative_responses = [(resp, conf) for resp, conf in response_confidence_pairs 
                                if conf <= negative_threshold]
            
            # Sort negatives by confidence (ascending) and take bottom 3
            negative_responses.sort(key=lambda x: x[1])
            result['bottom_negatives'] = [
                {'response': resp, 'confidence': conf} 
                for resp, conf in negative_responses[:3]
            ]
            
            filtered_results.append(result)
        
        return filtered_results

def main():
    """Main function to run the enhanced analysis"""
    # Initialize analyzer
    project_dir = Path(__file__).resolve().parents[2]
    processed_dir = project_dir / "data" / "processed"
    analyzer = EnhancedResponseAnalyzer(processed_dir / "processedchem.csv")
    
    # Load data
    if not analyzer.load_data():
        print("Failed to load data. Please check the file path.")
        return None, None
    
    # Analyze confidence distribution with enhanced metrics
    stats, confidence_array = analyzer.analyze_confidence_distribution(
        positive_threshold=0.7, 
        negative_threshold=0.3
    )
    
    # if stats is not None:
    #     # Filter responses
    #     filtered_results = analyzer.filter_responses(
    #         positive_threshold=0.7, 
    #         negative_threshold=0.3
    #     )
        
    #     return analyzer, filtered_results
    
    return analyzer, None

if __name__ == "__main__":
    analyzer, results = main()
