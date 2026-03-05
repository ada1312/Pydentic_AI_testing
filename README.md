# DBT Schema Quality Assessment Tool

AI-powered tool for evaluating and scoring dbt schema documentation quality using local LLMs (Ollama) or cloud AI providers.

## What This Tool Does

This tool automatically assesses the quality of your dbt schema descriptions (models and columns) and provides:

- **Quality Scores** (0-100) for each model and column description
- **Issue Detection** categorized by severity (CRITICAL, HIGH, MEDIUM, LOW)
- **Actionable Suggestions** for improving documentation
- **Comprehensive Metrics** including execution time, score distribution, and issue counts
- **Flexible Output** in human-readable format or structured JSON for cloud deployment

### Assessment Criteria

The tool evaluates descriptions based on:

- **CRITICAL Issues** (0-40 points):
  - Missing descriptions
  - Placeholder text (TODO, TBD, N/A, etc.)
  
- **HIGH Issues** (41-60 points):
  - Too short (< 20 characters)
  - Missing context or overly generic
  
- **MEDIUM Issues** (61-89 points):
  - Too long (> 300 characters)
  - Vague words (data, value, info, etc.)
  - Redundant phrases ("this column", "contains the")
  - Formatting issues
  
- **LOW Issues** (90-99 points):
  - Missing capitalization
  - Missing ending punctuation

## Installation

### Prerequisites

- Python 3.10 or higher
- Ollama installed and running (for local AI assessment)

### Setup

1. **Install Ollama** (for local AI):
   ```bash
   brew install ollama
   ```

2. **Pull the Qwen 2.5 model**:
   ```bash
   ollama pull qwen2.5
   ```

3. **Start Ollama server**:
   ```bash
   ollama serve
   ```

4. **Install Python dependencies**:
   ```bash
   pip install pydantic pydantic-ai pyyaml python-dotenv
   ```

## Usage

### Basic Usage (Human-Readable Output)

Assess a dbt schema file with default human-readable output:

```bash
python src/testing_pydantic.py sample_schema.yml
```

**Output:**
```
🎯 SCHEMA QUALITY REPORT
================================================================================

📊 Summary Statistics:
  Total items: 8 (models: 2, columns: 6)
  Average score: 55.62/100
  Execution time: 57.053s
  Issues found: CRITICAL=3, HIGH=4, MEDIUM=6, LOW=8
  Score distribution:
    Excellent (90-100)     0 (  0.0%) 
    Good (70-89)           0 (  0.0%) 
    Fair (50-69)           6 ( 75.0%) ███████████████
    Poor (0-49)            2 ( 25.0%) █████

🚨 Items Requiring Attention:

  ❌ CRITICAL (3 items):
    • stg_orders.order_id       Score: 65 → Placeholder text: 'na'
    • stg_orders.status         Score: 20 → Placeholder text like 'todo'
```

### JSON Output (Cloud-Ready)

Generate structured JSON output for APIs, cloud functions, or automation:

```bash
python src/testing_pydantic.py sample_schema.yml --json
```

### Save JSON to File

Export assessment results to a file for processing or storage:

```bash
python src/testing_pydantic.py sample_schema.yml --json --output report.json
```

### With Custom Documentation Files

Specify markdown documentation files explicitly:

```bash
python src/testing_pydantic.py sample_schema.yml --docs schema_docs.md --docs extra_docs.md
```

### Verbose Logging

Enable detailed logging for debugging:

```bash
python src/testing_pydantic.py sample_schema.yml --verbose
```

## Command-Line Options

```
usage: testing_pydantic.py [-h] [--docs DOCS] [--json] [--output OUTPUT] 
                           [--verbose] schema

positional arguments:
  schema           Path to dbt schema.yml file

optional arguments:
  -h, --help       Show help message and exit
  --docs DOCS      Path to dbt docs .md file (repeatable)
  --json           Output structured JSON instead of human-readable format
  --output OUTPUT  Write JSON output to file (requires --json)
  --verbose        Enable verbose logging
```

## Output Formats

### Human-Readable Format (Default)

Shows a comprehensive report with:
- Executive summary with statistics
- Score distribution chart
- Items grouped by severity
- Detailed ratings for models and columns
- Issue descriptions and improvement suggestions

### JSON Format (`--json`)

Structured output perfect for:
- Cloud APIs and serverless functions
- CI/CD pipelines
- Monitoring dashboards
- Data processing workflows

**Example JSON structure:**
```json
{
  "status": "success",
  "metrics": {
    "total_items": 8,
    "total_models": 2,
    "total_columns": 6,
    "average_score": 55.62,
    "excellent_count": 0,
    "good_count": 0,
    "fair_count": 6,
    "poor_count": 2,
    "critical_issues_count": 3,
    "high_issues_count": 4,
    "medium_issues_count": 6,
    "low_issues_count": 8,
    "execution_time_seconds": 57.053,
    "timestamp": "2026-03-05T22:51:09Z"
  },
  "results": [
    {
      "type": "model",
      "name": "stg_orders",
      "score": 65,
      "rating": "C",
      "issues": [
        {
          "severity": "HIGH",
          "message": "Too short",
          "deduction": 40
        }
      ],
      "suggestions": [
        "Expand description with specific details"
      ]
    }
  ]
}
```

## Exit Codes

The tool returns different exit codes for automation:

- `0` - Success, quality meets standards (avg score ≥50, no critical issues)
- `1` - Success, but quality issues detected
- `2` - Error (file not found, AI unavailable, etc.)

## Features

### AI-Powered Assessment

Uses Ollama (local) or cloud AI providers via Pydantic AI:
- **Intelligent evaluation** beyond simple rule-based checks
- **Context-aware suggestions** tailored to your descriptions
- **Consistent scoring** across all documentation

### Documentation Resolution

Automatically resolves dbt `{{ doc('name') }}` references:
- Loads markdown documentation from `.md` files
- Validates doc references exist
- Reports missing documentation blocks

### Performance Metrics

Tracks and reports:
- Total execution time
- Assessment throughput
- Issue distribution
- Quality trends over time

### Cloud-Ready Architecture

Built for deployment to:
- AWS Lambda
- Google Cloud Functions
- Azure Functions
- FastAPI/Flask APIs
- CI/CD pipelines

See [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for details.

## Example Workflow

### 1. Local Development

```bash
# Assess your schema
python src/testing_pydantic.py schema.yml

# Fix critical issues
# ... edit schema.yml ...

# Re-assess to verify improvements
python src/testing_pydantic.py schema.yml
```

### 2. CI/CD Pipeline

```yaml
# .github/workflows/schema-quality.yml
- name: Assess Schema Quality
  run: |
    python src/testing_pydantic.py schema.yml --json --output report.json
    
- name: Check Quality Gate
  run: |
    avg_score=$(jq '.metrics.average_score' report.json)
    if (( $(echo "$avg_score < 70" | bc -l) )); then
      echo "Quality gate failed: $avg_score < 70"
      exit 1
    fi
```

### 3. Cloud Deployment

```python
# AWS Lambda function
import json
from pathlib import Path
from testing_pydantic import assess_schema, load_schema, load_docs

def lambda_handler(event, context):
    schema = load_schema(Path(event['schema_path']))
    docs = load_docs([])
    
    report = assess_schema(schema, docs)
    
    return {
        'statusCode': 200 if report.status == 'success' else 500,
        'body': report.model_dump_json()
    }
```

## File Structure

```
.
├── src/
│   └── testing_pydantic.py    # Main assessment tool
├── sample_schema.yml          # Example dbt schema file
├── schema_docs.md             # Example documentation
├── CLOUD_DEPLOYMENT.md        # Cloud deployment guide
└── README.md                  # This file
```

## Advanced Usage

### Multi-Model Optimization

For cloud deployments, implement cascading assessment:

```python
# Use fast model first
quick_result = assess_with_model("qwen2.5")

# Only use expensive model for low-confidence results
if quick_result.score < 80:
    refined_result = assess_with_model("gpt-4")
    return refined_result
    
return quick_result
```

### Caching for Performance

Implement Redis caching to avoid re-processing:

```python
import hashlib
import redis

cache = redis.Redis()

def assess_with_cache(text: str):
    key = f"assessment:{hashlib.sha256(text.encode()).hexdigest()}"
    cached = cache.get(key)
    
    if cached:
        return json.loads(cached)
    
    result = assess_description(text)
    cache.setex(key, 3600, result.model_dump_json())
    
    return result
```

## Troubleshooting

### Ollama Connection Issues

**Error:** `AI assessment unavailable - ensure Ollama is running`

**Solution:**
```bash
# Check if Ollama is running
lsof -i :11434

# If not running, start it
ollama serve

# In another terminal, verify the model is installed
ollama list
ollama pull qwen2.5
```

### Slow Assessment Speed

**Issue:** Assessment takes too long

**Solutions:**
- Use a faster model: `ollama pull qwen2.5:1.5b` (smaller, faster)
- Enable caching for repeated assessments
- Use parallel processing for multiple schemas
- Consider cloud deployment with more resources

### API Rate Limits (Cloud)

**Issue:** Rate limit errors with cloud AI providers

**Solutions:**
- Implement exponential backoff retry logic
- Use request batching when supported
- Add caching layer to reduce API calls
- Consider using reserved capacity/dedicated endpoints

## Contributing

Contributions welcome! Areas for improvement:
- Support for additional AI providers
- Custom scoring rules and weights
- Batch processing for multiple schemas
- GitHub Copilot skill integration
- Automated fix suggestions

## License

MIT License - see LICENSE file for details

## Support

For issues or questions:
- Open an issue on GitHub
- Check [CLOUD_DEPLOYMENT.md](CLOUD_DEPLOYMENT.md) for cloud-specific guidance
- Review example outputs in `test outputs/` directory
