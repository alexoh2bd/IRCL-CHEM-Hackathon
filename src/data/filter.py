import pandas as pd
import json
import ast
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
import torch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PassageCandidate:
    """Represents a generated passage candidate for filtering"""
    id: int
    title: str
    passage: str
    label: str  # 'positive' or 'negative'
    neg_type: Optional[str]
    explanation: Optional[str]
    confidence: float

class RelevanceFilter:
    """
    Filters generated passages based on query-instruction relevance
    Following the methodology from Promptriever paper
    """
    
    def __init__(self, model_name: str = "jhu-clsp/FollowIR-7B"):
        """
        Initialize the relevance filter
        
        Args:
            model_name: Name of the model to use for filtering (e.g., "followir-7b")
        """
        self.model_name = model_name
        self.filter_model = self.load_filter_model(model_name)
        
    def load_filter_model(self, model_name):
        """
        Load the filtering model (FollowIR-7B or equivalent) with memory optimization
        """
        logger.info(f"Loading filter model: {model_name}")
        project_dir = Path(__file__).resolve().parents[2]
        model_dir = project_dir / "models"
        model_dir.mkdir(exist_ok=True)
        
        try:
            # Check available memory and GPU
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"GPU memory available: {gpu_memory:.1f} GB")
            
            # Load with memory optimizations
            logger.info("Loading model with memory optimizations...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,  # Use half precision
                device_map="auto",          # Automatic device placement
                low_cpu_mem_usage=True,     # Reduce CPU memory usage
                trust_remote_code=True      # Allow custom code if needed
            )
            
            logger.info("Model loaded successfully, saving to local directory...")
            model.save_pretrained(model_dir)

            tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
            
            tokenizer.save_pretrained(model_dir)

            logger.info("Model and tokenizer saved successfully")
            
            # Clear memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'template': "Is the passage about {query}?: {text}",
                'token_true_id': 0,
                'token_false_id': 1
            }
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory. Try using CPU-only loading.")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            # Try CPU-only fallback
            logger.info("Attempting CPU-only loading...")
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="cpu",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True
                )
                model.save_pretrained(model_dir)
                
                tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.padding_side = "left"
                tokenizer.save_pretrained(model_dir)
                
                logger.info("Model saved successfully using CPU fallback")
                del model
                
                return {
                    'model': model,
                    'tokenizer': tokenizer,
                    'template': "Is the passage about {query}?: {text}",
                    'token_true_id': 0,
                    'token_false_id': 1
                }
                
            except Exception as fallback_error:
                logger.error(f"CPU fallback also failed: {fallback_error}")
                raise
                
    def _predict_relevance(self, query: str, passage: str) -> Tuple[bool, float]:
        """
        Use FollowIR-7B model to predict relevance between query and passage
        
        Args:
            query: The query text
            passage: The passage text
            
        Returns:
            Tuple of (is_relevant, confidence_score)
        """
        # Format the prompt using the template
        prompt = self.filter_model['template'].format(query=query, text=passage)
        
        # Tokenize the input
        inputs = self.filter_model['tokenizer'](
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=2048
        ).to('cuda')
        
        # Generate prediction
        with torch.no_grad():
            outputs = self.filter_model['model'](**inputs)
            logits = outputs.logits[0, -1, :]  # Get logits for the last token
            
            # Get probabilities for true/false tokens
            true_logit = logits[self.filter_model['token_true_id']]
            false_logit = logits[self.filter_model['token_false_id']]
            
            # Apply softmax to get probabilities
            true_prob = torch.softmax(torch.stack([true_logit, false_logit]), dim=0)[0]
            false_prob = torch.softmax(torch.stack([true_logit, false_logit]), dim=0)[1]
            
            # Determine relevance and confidence
            is_relevant = true_prob > false_prob
            confidence = max(true_prob, false_prob).item()
            
        return is_relevant.item(), confidence
    
    def check_instruction_relevance(self, query: str, instruction: str, passage: str) -> Tuple[bool, float]:
        """
        Check if a passage is relevant to the query+instruction combination
        
        Args:
            query: The original query
            instruction: The instruction defining relevance criteria  
            passage: The passage to evaluate
            
        Returns:
            Tuple of (is_relevant, confidence_score)
        """
        # Combine query and instruction for relevance checking
        combined_query = f"{query} {instruction}"
        
        return self._predict_relevance(combined_query, passage)
    
    def check_query_relevance(self, query: str, passage: str) -> Tuple[bool, float]:
        """
        Check if a passage is relevant to just the query (without instruction)
        
        Args:
            query: The original query
            passage: The passage to evaluate
            
        Returns:
            Tuple of (is_relevant, confidence_score)
        """
        return self._predict_relevance(query, passage)
    
    def parse_response_json(self, response_str: str) -> List[PassageCandidate]:
        """
        Parse the JSON response containing generated passages
        
        Args:
            response_str: JSON string containing passage candidates
            
        Returns:
            List of PassageCandidate objects
        """
        try:
            # Handle both JSON string and Python literal evaluation
            if isinstance(response_str, str):
                try:
                    data = json.loads(response_str)
                except json.JSONDecodeError:
                    # Try literal_eval for Python-like strings
                    data = ast.literal_eval(response_str)
            else:
                data = response_str
            
            candidates = []
            for item in data:
                candidate = PassageCandidate(
                    id=item.get('id', 0),
                    title=item.get('title', ''),
                    passage=item.get('passage', ''),
                    label=item.get('label', ''),
                    neg_type=item.get('neg_type'),
                    explanation=item.get('explanation'),
                    confidence=item.get('confidence', 0.0)
                )
                candidates.append(candidate)
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error parsing response JSON: {e}")
            return []
    
    def filter_candidates(self, query: str, instruction: str, candidates: List[PassageCandidate]) -> Dict[str, List[PassageCandidate]]:
        """
        Filter passage candidates based on relevance criteria
        
        Following paper methodology:
        1. Check instruction negatives are actually instruction-negative
        2. Check instruction positives are actually relevant
        
        Args:
            query: The original query
            instruction: The instruction defining relevance
            candidates: List of passage candidates to filter
            
        Returns:
            Dictionary with filtered candidates categorized by type
        """
        filtered_results = {
            'valid_positives': [],
            'valid_negatives': [],
            'discarded_positives': [],
            'discarded_negatives': []
        }
        
        for candidate in candidates:
            query_relevant, query_score = self.check_query_relevance(query, candidate.passage)
            instruction_relevant, instruction_score = self.check_instruction_relevance(
                query, instruction, candidate.passage
            )
            
            if candidate.label == 'positive':
                # For positives: must be relevant to both query AND instruction
                if query_relevant and instruction_relevant:
                    filtered_results['valid_positives'].append(candidate)
                    logger.info(f"Kept positive candidate {candidate.id}: Q={query_score:.2f}, I={instruction_score:.2f}")
                else:
                    filtered_results['discarded_positives'].append(candidate)
                    logger.info(f"Discarded positive candidate {candidate.id}: Q={query_score:.2f}, I={instruction_score:.2f}")
                    
            elif candidate.label == 'negative':
                # For negatives: must be relevant to query but NOT to instruction
                if query_relevant and not instruction_relevant:
                    filtered_results['valid_negatives'].append(candidate)
                    logger.info(f"Kept negative candidate {candidate.id}: Q={query_score:.2f}, I={instruction_score:.2f}")
                else:
                    filtered_results['discarded_negatives'].append(candidate)
                    logger.info(f"Discarded negative candidate {candidate.id}: Q={query_score:.2f}, I={instruction_score:.2f}")
        
        return filtered_results

def process_dataset(input_path: str, output_path: str, filter_model: str = "followir-7b"):
    """
    Process the entire dataset and filter passage candidates
    
    Args:
        csv_path: Path to input CSV file
        output_path: Path for filtered output CSV
        filter_model: Model name for filtering
    """
    # Load dataset
    logger.info(f"Loading dataset from {input_path}")
    project_dir = Path(__file__).resolve().parents[2]
    processed_dir = project_dir / "data" / "processed"
    df = pd.read_csv(processed_dir / input_path)
    
    # Initialize filter
    relevance_filter = RelevanceFilter(filter_model)
    
    # Process each row
    filtered_data = []
    
    for idx, row in df.iterrows()[:1]:
        logger.info(f"Processing row {idx}/{len(df)}")
        
        # Parse candidates from Response column
        candidates = relevance_filter.parse_response_json(row['Response'])
        
        if not candidates:
            logger.warning(f"No valid candidates found in row {idx}")
            continue
        
        # Filter candidates
        filtered_results = relevance_filter.filter_candidates(
            row['Query'], 
            row['Instruction'], 
            candidates
        )
        
        # Create output records
        for pos_candidate in filtered_results['valid_positives']:
            filtered_data.append({
                'original_id': row['ID'],
                'query': row['Query'],
                'instruction': row['Instruction'],
                'passage_id': pos_candidate.id,
                'title': pos_candidate.title,
                'passage': pos_candidate.passage,
                'label': 'positive',
                'filter_status': 'kept',
                'neg_type': pos_candidate.neg_type,
                'explanation': pos_candidate.explanation,
                'confidence': pos_candidate.confidence
            })
        
        for neg_candidate in filtered_results['valid_negatives']:
            filtered_data.append({
                'original_id': row['ID'],
                'query': row['Query'],
                'instruction': row['Instruction'],
                'passage_id': neg_candidate.id,
                'title': neg_candidate.title,
                'passage': neg_candidate.passage,
                'label': 'negative',
                'filter_status': 'kept',
                'neg_type': neg_candidate.neg_type,
                'explanation': neg_candidate.explanation,
                'confidence': neg_candidate.confidence
            })
    
    # Save filtered results
    filtered_df = pd.DataFrame(filtered_data)
    filtered_df.to_csv(processed_dir / output_path, index=False)
    logger.info(f"Saved {len(filtered_df)} filtered passages to {output_path}")
    
    # Print summary statistics
    print("\n=== Filtering Summary ===")
    print(f"Total passages processed: {len(filtered_df)}")
    print(f"Valid positives: {len(filtered_df[filtered_df['label'] == 'positive'])}")
    print(f"Valid negatives: {len(filtered_df[filtered_df['label'] == 'negative'])}")

# Example usage
if __name__ == "__main__":
    model_name = "jhu-clsp/FollowIR-7B"
    # Process your dataset
    # load_filter_model(model_name)
    process_dataset(
        input_path="processedchem.csv",
        output_path="filtered_passages.csv",
        filter_model=model_name
    )