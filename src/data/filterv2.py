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

class ResponseAnalyzer:
    def __init__(self, csv_path: str):
        """
        Initialize the ResponseAnalyzer with the path to processedchem.csv
        
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
    
   
    def extract_confidence_scores(self, responses: List[Dict]) -> List[float]:
        """
        Extract confidence scores from response objects
        
        Args:
            responses (List[Dict]): List of response objects
            
        Returns:
            List[float]: List of confidence scores
        """
        confidence_scores = []
        
        for response in responses:
            if isinstance(response, dict):
                # Look for common confidence score keys
                confidence_keys = ['confidence', 'score', 'confidence_score', 'probability', 'relevance']
                
                for key in confidence_keys:
                    if key in response:
                        try:
                            score = float(response[key])
                            confidence_scores.append(score)
                            break
                        except (ValueError, TypeError):
                            continue
                            
        return confidence_scores
    
    def analyze_confidence_distribution(self):
        """Analyze the distribution of response confidence levels"""
        if self.df is None:
            print("Data not loaded. Please run load_data() first.")
            return
            
        print("Analyzing confidence score distribution...")
        all_confidence_scores = []
        response_counts = []
        
        # for idx, row in next(self.df.iterrows()):
        _, row = next(self.df.iterrows())
        responses = ast.literal_eval(row['Response'])
        confidence_scores = self.extract_confidence_scores(responses)
        
        all_confidence_scores.extend(confidence_scores)
        response_counts.append(len(responses))
        
        # Store detailed data for each row
        self.response_confidence_data.append({
            'ID': row['ID'],
            'Query': row['Query'],
            'num_responses': len(responses),
            'confidence_scores': confidence_scores,
            'responses': responses
        })
    
        # Convert to numpy array for analysis
        confidence_array = np.array(all_confidence_scores)
        
        if len(confidence_array) == 0:
            print("No confidence scores found in the data.")
            return
            
        # Calculate statistics
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
        
        print("\n=== CONFIDENCE SCORE DISTRIBUTION ANALYSIS ===")
        print(f"Total responses analyzed: {stats['total_responses']}")
        print(f"Mean confidence: {stats['mean_confidence']:.4f}")
        print(f"Median confidence: {stats['median_confidence']:.4f}")
        print(f"Standard deviation: {stats['std_confidence']:.4f}")
        print(f"Min confidence: {stats['min_confidence']:.4f}")
        print(f"Max confidence: {stats['max_confidence']:.4f}")
        print(f"25th percentile: {stats['q25']:.4f}")
        print(f"75th percentile: {stats['q75']:.4f}")
        
        # Response count statistics
        response_count_array = np.array(response_counts)
        print(f"\nAverage responses per query: {np.mean(response_count_array):.2f}")
        print(f"Max responses per query: {np.max(response_count_array)}")
        print(f"Min responses per query: {np.min(response_count_array)}")
        
        return stats, confidence_array
    
    def plot_confidence_distribution(self, confidence_array: np.ndarray):
        """Create visualizations of confidence score distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        axes[0, 0].hist(confidence_array, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].set_title('Distribution of Confidence Scores')
        axes[0, 0].set_xlabel('Confidence Score')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(confidence_array)
        axes[0, 1].set_title('Box Plot of Confidence Scores')
        axes[0, 1].set_ylabel('Confidence Score')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cumulative distribution
        sorted_scores = np.sort(confidence_array)
        cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
        axes[1, 0].plot(sorted_scores, cumulative, linewidth=2)
        axes[1, 0].set_title('Cumulative Distribution of Confidence Scores')
        axes[1, 0].set_xlabel('Confidence Score')
        axes[1, 0].set_ylabel('Cumulative Probability')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Density plot
        axes[1, 1].hist(confidence_array, bins=50, density=True, alpha=0.7, color='lightcoral')
        axes[1, 1].set_title('Density Plot of Confidence Scores')
        axes[1, 1].set_xlabel('Confidence Score')
        axes[1, 1].set_ylabel('Density')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
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
    
    def print_filtering_summary(self, filtered_results: List[Dict], 
                              positive_threshold: float, negative_threshold: float):
        """Print summary of filtering results"""
        total_queries = len(filtered_results)
        queries_with_positive = sum(1 for r in filtered_results if r['top_positive'] is not None)
        queries_with_negatives = sum(1 for r in filtered_results if len(r['bottom_negatives']) > 0)
        
        print(f"\n=== FILTERING RESULTS SUMMARY ===")
        print(f"Positive threshold: {positive_threshold}")
        print(f"Negative threshold: {negative_threshold}")
        print(f"Total queries: {total_queries}")
        print(f"Queries with positive responses above threshold: {queries_with_positive} ({queries_with_positive/total_queries*100:.1f}%)")
        print(f"Queries with negative responses below threshold: {queries_with_negatives} ({queries_with_negatives/total_queries*100:.1f}%)")
        
        # Show examples
        print(f"\n=== EXAMPLES ===")
        for i, result in enumerate(filtered_results[:3]):  # Show first 3 examples
            print(f"\nQuery {i+1}: {result['Query'][:100]}...")
            if result['top_positive']:
                print(f"  Top positive (confidence: {result['top_positive']['confidence']:.3f})")
            else:
                print(f"  No positive responses above threshold")
                
            if result['bottom_negatives']:
                print(f"  Bottom negatives: {len(result['bottom_negatives'])} responses")
                for j, neg in enumerate(result['bottom_negatives']):
                    print(f"    {j+1}. Confidence: {neg['confidence']:.3f}")
            else:
                print(f"  No negative responses below threshold")

def main():
    """Main function to run the analysis"""
    # Initialize analyzer
    project_dir = Path(__file__).resolve().parents[2]
    processed_dir = project_dir / "data" / "processed"
    analyzer = ResponseAnalyzer(processed_dir/"processedchem.csv")
    
    # Load data
    if not analyzer.load_data():
        print("Failed to load data. Please check the file path.")
        return
    
    # Analyze confidence distribution
    # stats, confidence_array = analyzer.analyze_confidence_distribution()
    
    # if confidence_array is not None and len(confidence_array) > 0:
    #     # Plot distribution
    #     analyzer.plot_confidence_distribution(confidence_array)
        
    #     # Filter responses with different thresholds
    #     positive_threshold = 0.7  # Adjust based on your data
    #     negative_threshold = 0.3  # Adjust based on your data
        
    #     filtered_results = analyzer.filter_responses(positive_threshold, negative_threshold)
    #     analyzer.print_filtering_summary(filtered_results, positive_threshold, negative_threshold)
        
    #     return analyzer, filtered_results
    
    # return analyzer, None

if __name__ == "__main__":
    analyzer, results = main()