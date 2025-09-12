# ChemLit-QA Instruction-Following Retrieval Dataset

## Overview

This project creates an instruction-following retrieval dataset for chemistry domain using the ChemLit-QA dataset as a foundation. Inspired by the **Promptriever** methodology, we generate instruction-positive and instruction-negative passage pairs to train retrieval models that can follow nuanced relevance criteria beyond simple query matching.

## Motivation & Reasoning

### The Problem
Traditional retrieval systems operate on basic query-document similarity, but real-world information needs often require more nuanced relevance criteria. For example, a chemistry researcher might need passages that not only discuss "polymer crystallization" but specifically include "temperature ranges and kinetic parameters."

### Our Approach
Following the Promptriever framework, we:
1. **Generate Instructions**: Create natural language instructions that narrow relevance definitions
2. **Generate Passage Candidates**: Produce both instruction-positive and instruction-negative examples
3. **Filter with FollowIR**: Use FollowIR-7B model to validate passage relevance
4. **Create Training Data**: Build high-quality instruction-following retrieval datasets

## Dataset Information

**Base Dataset**: [ChemLit-QA](https://github.com/geemi725/ChemLit-QA)
- 1,000+ expert-validated chemistry Q&A pairs
- Covers diverse chemistry topics with rigorous expert evaluation
- Provides rich context for generating nuanced instructions

**Generated Dataset Structure**:
```
Row_Index
Query
OgPositive
Instruction
Response
- 
```

## Methodology

### Step 1: Instruction Generation
Using `text.yaml` template with Cerebras Qwen-3-32B model:
- **Input**: Chemistry question + answer pair
- **Output**: Natural language instruction that adds qualifications
- **Styles**: Persona, negation, background, generic
- **Lengths**: Short, medium, long, very long


### Step 2: Passage Generation
Using `responses.yaml` template:
- Generate 1 instruction-positive passage
- Generate 5-6 instruction-negative passages
- Each negative follows specific failure modes:
  - Omission of required details
  - Different interpretation
  - Mentions excluded elements
  - Partial answers
  - Outdated information
  - Unsupported inference

### Step 3: Quality Filtering
Using FollowIR-7B model for validation:
- **Instruction Positives**: Must be relevant to both query AND instruction
- **Instruction Negatives**: Must be relevant to query but NOT to instruction
- Confidence scoring for each relevance decision
- Memory-efficient batch processing with checkpointing 

## Examples

Example 1:
```yaml
Original Query: What happens to the separation of compounds when the pressure is increased?
Original Response: Some compounds separate better at higher pressure
Generated Instruction: A relevant passage must describe how increased pressure chemically or physically influences the separation of compounds, specifically addressing the type of compounds involved (e.g., volatile, polar, or gas-phase species) and the separation mechanism (e.g., distillation, gas solubility, or chromatography). General statements about pressure effects without tying them to molecular interactions, phase behavior, or method-specific adjustments are not sufficient for relevance.
I-Positive Response: Increasing pressure affects compound separation differently depending on the method. In distillation, higher pressure raises boiling points for volatile compounds, making their separation less efficient due to reduced vapor-liquid equilibrium differences. For gas-phase compounds in gas chromatography, elevated column inlet pressure accelerates flow rates, shortening retention times and decreasing separation efficiency, while reducing outlet pressure improves resolution for non-polar compounds by prolonging interaction with the stationary phase. Polar compounds in liquid chromatography exhibit altered solubility at higher pressures, impacting partitioning between phases. Additionally, gas solubility in liquids increases with pressure (Henry’s Law), which complicates methods like pressure-sensitive extraction. Source chunks ID1-3 support these specific compound-type and mechanism-dependent effects.
I-Negative Response 1: Pressure influences chemical equilibria by favoring low-molar-volume reactions. In high-pressure environments, compounds often resist phase changes. This can stabilize reaction intermediates but may also hinder equilibration.
I-Negative Response 2: Elevated pressure reduces the volatility of compounds, narrowing boiling point differences. In vacuum distillations, lower pressure increases separation efficiency for volatile mixtures by lowering their boiling points.

Example 2:
```yaml
Original Query: Why was UV-ozone treatment used instead of the harsher piranha treatment for treating glass substrates?
Original Response: UV-ozone treatment was used instead of the harsher piranha treatment because it allowed pre-determining the film growth area by covering the undesired areas with acrylic tape, whereas the piranha treatment tends to etch away the ITO.
Generated Instruction: A relevant passage must explain the use of a surface treatment method that allows controlled application of film growth by utilizing masking techniques, and must highlight the importance of preserving functional materials on the substrate. Passages that only mention alternative treatments without addressing their specific impacts on substrate integrity or masking compatibility are not relevant. Additionally, exclude any responses that provide general descriptions of either method without focusing on the comparative advantages related to material protection and film area definition.
I-Positive Response: UV-ozone treatment was selected for glass substrates due to its compatibility with photolithography masking techniques and its ability to preserve functional materials during surface preparation. Unlike piranha etch, which employs a mixture of sulfuric acid and hydrogen peroxide at elevated temperatures, UV-ozone relies on reactive oxygen species generated by UV light to oxidize the glass surface. This method avoids contact with corrosive chemicals, preventing degradation of sensitive films like organic passivations or photoresists. Studies in C1 demonstrate that UV-ozone preserves native surface charge distributions critical for semiconductor patterning, whereas piranha etching (C2) often introduces microcracks and alters surface topography irreversibly. By selectively oxidizing exposed areas without requiring physical masking layers, UV-ozone enables accurate definition of film growth regions while maintaining substrate material integrity, as validated by C3's comparative analysis.
I-Negative Response 1: UV-ozone was used for glass substrates because it minimizes liquid chemical exposure compared to piranha etch. While piranha etching involves hazardous sulfuric acid and strong oxidizers, UV-ozone utilizes gaseous ozone and UV radiation, reducing chemical waste and operator risk. However, it does not specify how each method affects substrate material stability or compatibility with masking processes, nor does it address functional material preservation as required.
I-Negative Response 2: The choice of UV-ozone treatment over piranha etching was primarily motivated by operational cost savings. UV systems require no disposable chemicals and offer faster processing times, as demonstrated in C4. Although C5 notes that both methods achieve surface hydrophilicity, the focus on economical advantages shifts from the instruction’s emphasis on masking compatibility and material integrity preservation during film patterning.



## Configuration

```python
# Ingestion Settings
BATCH_SIZE = 20              # Rows to write at once
CHECKPOINT_INTERVAL = 5      # Save state every N rows
MAX_ROWS = None             # Limit processing (None = all)

# Model Settings
MODEL = "qwen-3-32b"        # Instruction generation model
FILTER_MODEL = "jhu-clsp/FollowIR-7B"  # Relevance filtering model
```

## Results & Outcomes

### Dataset Statistics
- **Source**: 1,000+ ChemLit-QA entries
- **Generated**: ~6,000+ instruction-passage pairs

- **Coverage**: Diverse chemistry domains with expert-validated foundations

### Quality Metrics
- **Instruction Diversity**: 4 styles × 4 lengths = 16 instruction variants
- **Failure Mode Coverage**: 6 distinct negative passage types

##### INSTRUCTION POSITIVE Prompt-Response Pairs 
- **Count**: 867 (17.2% of total)
- **Mean**: 0.9553
- **Median**: 0.9500
- **Std Dev**: 0.0259
- **Variance**: 0.0007
- **Range**: 0.8000 - 1.0000
- **IQR**: 0.9500 - 0.9800
- **Skewness**: -1.0970
- **Kurtosis**: 3.5059

##### INSTRUCTION NEGATIIVE Prompt-Response Pairs
- **Count**: 4174 (82.7% of total)
- **Mean**: 0.5290
- **Median**: 0.5700
- **Std Dev**: 0.1829
- **Variance**: 0.0334
- **Range**: 0.0000 - 1.0000
- **IQR**: 0.4000 - 0.6800
- **Skewness**: -0.3999
- **Kurtosis**: -0.7277


### Key Innovations
1. **Chemistry-Specific Instructions**: Domain-aware relevance criteria
2. **Automated Quality Control**: Model-based filtering pipeline
3. **Scalable Architecture**: Memory-efficient processing for large datasets
4. **Comprehensive Validation**: Multi-stage quality assurance

### Pitfalls and Improvements
1. **Inconsistent LM Responses and Data Loss**: Inconsistent response formatting led to loss of responses and pipeline issues. $903 / 1025$ entries logged. 
2. **Incomplete Implementation of the Promptriever Data Generation Workflow**: Intended to use the instruction retrieval encoder FollowIR to assess the relevance of prompt response pairs and filter and filter positive and negative responses out of range. 


## Usage

### Generate Dataset
```bash
# Run instruction-following dataset generation
python src/data/ingestion_v2.py

# Filter generated passages
python src/data/filterv2.py
```

### Resume from Checkpoint
The system automatically resumes from the last checkpoint if interrupted:
```bash
# Will resume from last saved state
python src/data/ingestion_v2.py
```

## File Structure

```
├── src/data/
│   ├── ingestion_v2.py      # Memory-efficient dataset generation
│   ├── filter.py            # FollowIR-based quality filtering
│   └── ingestion.py         # Legacy/testing code
├── data/
│   ├── raw/chemlitqa.csv    # Source ChemLit-QA dataset
│   └── processed/           # Generated datasets
├── assets/
│   └── Promptriever.pdf     # Methodology reference
├── text.yaml                # Instruction generation template
├── responses.yaml           # Passage generation template
└── requirements.txt         # Dependencies
```

## Dependencies

```txt
pandas
tqdm
python-dotenv
huggingface_hub
cerebras-cloud-sdk
datasets
jinja2
transformers
torch
accelerate
```

## Future Work

1. **Evaluation Framework**: Implement retrieval performance benchmarks
2. **Model Training**: Fine-tune retrieval models on generated dataset
3. **Domain Expansion**: Extend methodology to other scientific domains
4. **Interactive Filtering**: Human-in-the-loop quality validation

## Citation

This work builds upon:
- **ChemLit-QA**: Expert-validated chemistry Q&A dataset
- **Promptriever**: Instruction-following retrieval methodology
- **FollowIR**: Instruction-aware relevance modeling

## License

MIT License - See LICENSE file for details.

---

*Generated using Cerebras Cloud SDK and FollowIR-7B for high-quality instruction-following retrieval data in the chemistry domain.*
