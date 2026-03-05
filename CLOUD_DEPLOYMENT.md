# Cloud Deployment Guide

## Cloud-Ready Features

Your schema assessment tool is now optimized for cloud deployment with structured outputs, logging, and metrics tracking.

## Output Formats

### 1. JSON Output (Cloud/API Ready)
```bash
python testing_pydantic.py schema.yml --json
```

**Benefits:**
- Machine-readable structured data
- Easy to parse in cloud functions/APIs
- Includes comprehensive metrics
- Supports monitoring dashboards
- Compatible with logging services

### 2. File Output
```bash
python testing_pydantic.py schema.yml --json --output report.json
```

**Use Cases:**
- Save to cloud storage (S3, GCS, Azure Blob)
- Process in data pipelines
- Archive assessment history
- Feed into BI tools

### 3. Human-Readable (Default)
```bash
python testing_pydantic.py schema.yml
```

**Use Cases:**
- Local development
- CI/CD pipeline logs
- Manual review

## Exit Codes

- `0` - Success, good quality (avg score ≥50, no critical issues)
- `1` - Success but quality issues detected
- `2` - Error (file not found, AI unavailable, etc.)

## Structured Logging

All operations log structured messages:
```
2026-03-05 22:40:18,009 - __main__ - INFO - Starting schema assessment
2026-03-05 22:40:30,850 - __main__ - INFO - Assessment complete: 5 items in 12.54s
```

**Cloud Integration:**
- CloudWatch Logs (AWS)
- Cloud Logging (GCP)
- Azure Monitor
- DataDog, New Relic, etc.

## Metrics Tracking

Every assessment includes:
- **Performance**: execution_time_seconds, timestamp
- **Volume**: total_items, total_models, total_columns
- **Quality**: average_score, score distribution
- **Issues**: counts by severity (CRITICAL, HIGH, MEDIUM, LOW)

**Use for:**
- Performance monitoring
- Quality trend analysis
- SLA tracking
- Alerting on regression

## Cloud Deployment Patterns

### 1. AWS Lambda Function
```python
import json
from testing_pydantic import assess_schema, load_schema, load_docs

def lambda_handler(event, context):
    schema = load_schema(Path(event['schema_path']))
    docs = load_docs([Path(p) for p in event.get('docs_paths', [])])
    
    report = assess_schema(schema, docs)
    
    return {
        'statusCode': 200 if report.status == 'success' else 500,
        'body': report.model_dump_json()
    }
```

### 2. FastAPI Endpoint
```python
from fastapi import FastAPI, UploadFile
from testing_pydantic import assess_schema, DbtSchema
import yaml

app = FastAPI()

@app.post("/assess")
async def assess_schema_endpoint(file: UploadFile):
    content = await file.read()
    data = yaml.safe_load(content)
    schema = DbtSchema.model_validate(data)
    
    report = assess_schema(schema, {})
    return report.model_dump()
```

### 3. GitHub Actions CI/CD
```yaml
- name: Assess Schema Quality
  run: |
    python testing_pydantic.py schema.yml --json --output report.json
    
- name: Upload Report
  uses: actions/upload-artifact@v3
  with:
    name: quality-report
    path: report.json
    
- name: Check Quality Gate
  run: |
    avg_score=$(jq '.metrics.average_score' report.json)
    if (( $(echo "$avg_score < 70" | bc -l) )); then
      echo "Quality gate failed: $avg_score < 70"
      exit 1
    fi
```

## Multi-Model Optimization (Future)

To optimize outputs with multiple AI models:

### Pattern 1: Cascade (Fast → Precise)
```python
# Use fast model first
quick_assessment = assess_with_model("qwen2.5")

# Only use expensive model if confidence is low
if quick_assessment.score < 80:
    refined = assess_with_model("gpt-4")
    return refined
return quick_assessment
```

### Pattern 2: Validator Model
```python
# Main assessment
assessment = assess_description(text)

# Critic model validates
validation = critic_model.validate(assessment)

if validation.confidence < 0.8:
    # Retry with different prompt/temperature
    assessment = retry_assessment(text, temperature=0.9)

return assessment
```

### Pattern 3: Model Routing
```python
def smart_assess(text: str):
    complexity = estimate_complexity(text)
    
    if complexity < 0.3:
        return assess_with_model("qwen2.5")  # Fast, cheap
    elif complexity < 0.7:
        return assess_with_model("llama3.3")  # Balanced
    else:
        return assess_with_model("gpt-4")  # Complex cases
```

## Caching Strategy

```python
import hashlib
import redis

cache = redis.Redis()

def assess_with_cache(text: str):
    # Generate cache key
    key = f"assessment:{hashlib.sha256(text.encode()).hexdigest()}"
    
    # Check cache
    cached = cache.get(key)
    if cached:
        return json.loads(cached)
    
    # Assess and cache
    result = assess_description(text)
    cache.setex(key, 3600, result.model_dump_json())  # 1 hour TTL
    
    return result
```

## Monitoring & Alerts

### CloudWatch Metrics (AWS)
```python
import boto3

cloudwatch = boto3.client('cloudwatch')

def publish_metrics(report):
    cloudwatch.put_metric_data(
        Namespace='SchemaQuality',
        MetricData=[
            {
                'MetricName': 'AverageScore',
                'Value': report.metrics.average_score,
                'Unit': 'None'
            },
            {
                'MetricName': 'CriticalIssues',
                'Value': report.metrics.critical_issues_count,
                'Unit': 'Count'
            }
        ]
    )
```

### Alerting Rule
```
IF critical_issues_count > 0 OR average_score < 60
THEN send_alert("Schema quality degraded")
```

## Cost Optimization

1. **Batch Processing** - Assess multiple schemas in single run
2. **Caching** - Cache identical descriptions (dedupe)
3. **Model Selection** - Use cheaper models for simple cases
4. **Rate Limiting** - Prevent runaway costs
5. **Async Processing** - Don't block on AI calls

## Example Cloud Response

```json
{
  "status": "success",
  "metrics": {
    "average_score": 78.5,
    "execution_time_seconds": 3.421,
    "critical_issues_count": 0,
    "timestamp": "2026-03-05T22:40:00Z"
  },
  "summary": "Assessment completed successfully. 10 items assessed with average quality score of 78.5/100."
}
```
