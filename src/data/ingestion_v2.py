# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import pandas as pd
from cerebras.cloud.sdk import Cerebras
from jinja2 import Template
import os
from tqdm import tqdm
import random
import json
import time
from typing import Dict, List, Optional
import pickle

# System prompts
isystem_prompt = "You are an expert chemistry data engineer. Given a chemistry question and a candidate positive passage, generate a *single natural-language instruction* that narrows the relevance definition so that ONLY passages which satisfy a specific, testable chemical requirement remain relevant. The instruction must:- add extra qualifications (e.g., ask for explicit monomer feed ratios, solvent system & casting method, reagent grades, temperature/time, measured yields, or safety handling),- *not* include the answer content (do not leak the passage's factual answer),- be written in natural free-form language,- follow the requested length and style (short/medium/long/very long) and style tag (persona, negation, background, or generic).Output JSON only: {'instruction':'...','length':'short|medium|long|very long','style':'persona|negation|background|generic'} "
rsystem_prompt = "You are an expert dataset generator for instruction-trained retrieval models. Your job is to produce **one instruction-positive passage** and **five or six instruction-negative passages** given a (QUERY, INSTRUCTION) pair. Follow these rules strictly."

class MemoryEfficientIngestion:
    def __init__(self, project_dir: Path, batch_size: int = 10, checkpoint_interval: int = 5):
        self.project_dir = project_dir
        self.data_dir = project_dir / 'data' 
        self.batch_size = batch_size
        self.checkpoint_interval = checkpoint_interval
        
        # Setup logging first
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Initialize client
        self.client = Cerebras(api_key=os.environ.get("CEREBRAS_API_KEY_HCKTHON"))
        self.model = 'qwen-3-32b'
        
        # State tracking
        self.checkpoint_file = self.data_dir / 'ingestion_checkpoint.pkl'
        self.output_file = self.data_dir / 'processed'/ 'processedchem.csv'
        self.state = self._load_checkpoint()
        
    def _load_checkpoint(self) -> Dict:
        """Load processing state from checkpoint file"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'rb') as f:
                    state = pickle.load(f)
                self.logger.info(f"Resumed from checkpoint: processed {state['processed_count']} rows")
                return state
            except Exception as e:
                self.logger.warning(f"Could not load checkpoint: {e}")
        
        return {
            'processed_count': 0,
            'last_processed_id': None,
            'total_rows': 0,
            'start_time': time.time()
        }
    
    def _save_checkpoint(self):
        """Save current processing state"""
        try:
            with open(self.checkpoint_file, 'wb') as f:
                pickle.dump(self.state, f)
            self.logger.info(f"Checkpoint saved: {self.state['processed_count']} rows processed")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {e}")
    
    def _append_to_csv(self, rows: List[Dict]):
        """Append rows to CSV file incrementally"""
        df_batch = pd.DataFrame(rows)
        
        # Write header only if file doesn't exist
        write_header = not self.output_file.exists()
        
        try:
            df_batch.to_csv(self.output_file, mode='a', header=write_header, index=False)
            self.logger.info(f"Appended {len(rows)} rows to {self.output_file}")
        except Exception as e:
            self.logger.error(f"Failed to write to CSV: {e}")
            raise
    
    def build_iprompt(self, question: str, response: str) -> str:
        """Build instruction prompt"""
        with open(self.project_dir / "instruction.yaml") as f:
            raw_text = f.read()
        
        styles = ["persona", "negation", "background", "generic"]
        lengths = ["short", "medium", "long", "very long"]
        variables = {
            "Question": question,
            "Response": response,
            "STYLE": random.choice(styles),
            "LENGTH": random.choice(lengths),
        }
        return Template(raw_text).render(variables)
    
    def build_rprompt(self, query: str, instruction: str) -> str:
        """Build response prompt"""
        with open(self.project_dir / "responses.yaml") as f:
            raw_text = f.read()
        
        variables = {
            "Query": query,
            "Instruction": instruction,
        }
        return Template(raw_text).render(variables)
    
    def process_single_row(self, row: pd.Series) -> List[Dict]:
        """Process a single row and return generated data"""
        try:
            # Step 1: Generate instruction]            assert row["Question"] is not None and row["Answer"] is not None

            prompt = self.build_iprompt(row['Question'], row['Answer'])
            ichat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": prompt},
                    {"role": "system", "content": isystem_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"}
            )
            
            instruction = json.loads(ichat_completion.choices[0].message.content)['instruction']
            # Step 2: Generate responses
            rprompt = self.build_rprompt(row["Question"], instruction)
            rchat_completion = self.client.chat.completions.create(
                messages=[
                    {"role": "user", "content": rprompt},
                    {"role": "system", "content": rsystem_prompt}
                ],
                model=self.model,
                response_format={"type": "json_object"}
            )
            
            # Process all response choices
            results = []
            for choice in rchat_completion.choices:
                try:
                    response = json.loads(choice.message.content)
                    # print("response", response['examples'])
                    assert type(response["examples"]) is list
                    result_row = {
                        "ID": row["ID"],
                        "Query": row["Question"],
                        "OgPositive": row["Answer"],
                        "Instruction": instruction,
                        "Response": response["examples"],
                    }
                    results.append(result_row)
                except json.JSONDecodeError as e:
                    self.logger.warning(f"Failed to parse response for ID {row['ID']}: {e}")
                    continue
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing row ID {row['ID']}: {e}\n ")
            return []
    
    def process_dataset(self, max_rows: Optional[int] = None):
        """Process the entire dataset with memory efficiency and checkpointing"""
        # Read CSV in chunks to avoid loading entire dataset
        chunk_size = self.batch_size * 10  # Read larger chunks for efficiency
        csv_file = self.data_dir / "raw" / "chemlitqa.csv"
        
        self.logger.info(f"Starting ingestion with batch_size={self.batch_size}, checkpoint_interval={self.checkpoint_interval}")
        
        # Get total row count for progress tracking
        if self.state['total_rows'] == 0:
            total_rows = sum(1 for _ in pd.read_csv(csv_file, chunksize=1000)) * 1000
            self.state['total_rows'] = min(total_rows, max_rows) if max_rows else total_rows
        
        processed_in_session = 0
        batch_buffer = []
        
        # Process CSV in chunks
        # for chunk in tqdm(pd.read_csv(csv_file, chunksize=chunk_size)):
        #     # Skip already processed rows
        #     if self.state['last_processed_id'] is not None:
        #         chunk = chunk[chunk['ID'] > self.state['last_processed_id']]
            
        #     # Apply max_rows limit
        #     if max_rows and self.state['processed_count'] >= max_rows:
        #         break
        df = pd.read_csv(csv_file)
        for _, row in df.iterrows():
            if max_rows and self.state['processed_count'] >= max_rows:
                break
            
            # Process single row
            assert type(row) is pd.Series
            results = self.process_single_row(row)
            batch_buffer.extend(results)
            
            self.state['processed_count'] += 1
            self.state['last_processed_id'] = row['ID']
            processed_in_session += 1
            
            # Write batch when buffer is full
            if len(batch_buffer) >= self.batch_size:
                self._append_to_csv(batch_buffer)
                batch_buffer = []
            
            # Save checkpoint periodically
            if processed_in_session % self.checkpoint_interval == 0:
                self._save_checkpoint()
            
            # Progress update
            if processed_in_session % 10 == 0:
                elapsed = time.time() - self.state['start_time']
                rate = self.state['processed_count'] / elapsed if elapsed > 0 else 0
                self.logger.info(f"Progress: {self.state['processed_count']}/{self.state['total_rows']} "
                                f"({rate:.2f} rows/sec)")
        
        # Write remaining batch
        if batch_buffer:
            self._append_to_csv(batch_buffer)
        
        # Final checkpoint
        self._save_checkpoint()
        
        elapsed = time.time() - self.state['start_time']
        self.logger.info(f"Ingestion completed! Processed {self.state['processed_count']} rows in {elapsed:.2f} seconds")
        
        # Clean up checkpoint file when done
        if self.checkpoint_file.exists():
            self.checkpoint_file.unlink()

if __name__ == '__main__':
    load_dotenv(find_dotenv())
    
    project_dir = Path(__file__).resolve().parents[2]
    
    # Configuration
    BATCH_SIZE = 20  # Number of rows to write at once
    CHECKPOINT_INTERVAL = 5  # Save checkpoint every N processed rows
    MAX_ROWS = None  # Set to limit processing (None for all rows)
    
    # Initialize and run ingestion
    ingestion = MemoryEfficientIngestion(
        project_dir=project_dir,
        batch_size=BATCH_SIZE,
        checkpoint_interval=CHECKPOINT_INTERVAL
    )
    
    try:
        ingestion.process_dataset(max_rows=MAX_ROWS)
    except KeyboardInterrupt:
        ingestion.logger.info("Process interrupted by user. Progress saved in checkpoint.")
    except Exception as e:
        ingestion.logger.error(f"Fatal error: {e}")
        raise
