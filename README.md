# Medical Diagnosis Extraction Demo

A practical demonstration of structured output generation from natural language medical texts using Google's Gemini AI API and Pydantic models.

## Overview

This project extracts structured medical diagnoses from unstructured clinical text, transforming narrative medical reports into JSON-formatted data with diagnostic terms, context, and temporal information.

## Features

- **Text Preprocessing**: Cleans and normalizes medical text files
- **AI-Powered Extraction**: Uses Gemini 2.5 Flash for intelligent diagnosis identification
- **Structured Output**: Returns JSON with term, context, and temporal aspects
- **Pydantic Validation**: Ensures data quality and type safety
- **Beautiful Terminal Output**: Color-coded results with emojis for better readability

## Project Structure

```
demo_basic_structured_output/
├── agents/
│   ├── __init__.py
│   └── agent.py           # Main extraction logic
├── preprocess/
│   ├── __init__.py
│   └── preprocess.py      # Text cleaning utilities
├── texts/
│   └── case1.txt          # Sample medical case
├── output/                # Generated JSON results
├── .env                   # API keys (not in repo)
├── pyproject.toml         # Dependencies
└── README.md
```

## Quick Start

### 1. Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- Google Gemini API key

### 2. Setup

```bash
# Clone or download the project
cd demo_basic_structured_output

# Install dependencies
uv sync

# Create environment file
cp .env.example .env  # or create manually
```

### 3. Configure API Key

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

> **Get your API key**: Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to generate a free Gemini API key.

### 4. Run the Demo

```bash
# Execute the extraction agent
uv run python agents/agent.py
```

## Expected Output

The demo will:

1. **Preprocess** the medical text from `texts/case1.txt`
2. **Extract** diagnoses using Gemini AI
3. **Display** results in the terminal with formatting
4. **Save** structured JSON to `output/diagnosis_extraction.json`

Example output structure:
```json
{
  "diagnostics": [
    {
      "term": "ventricular fibrillation",
      "context": "patient shocked for ventricular fibrillation",
      "temporal": "upon arrival at emergency room"
    }
  ]
}
```

## Usage with UV

```bash
# Install/update dependencies
uv sync

# Run the main script
uv run python agents/agent.py

# Add new dependencies
uv add package-name

# Run in development mode
uv run --dev python agents/agent.py

# Check installed packages
uv pip list
```

## Customization

### Adding New Medical Cases

1. Place text files in the `texts/` directory
2. Modify the `case_file` path in `agents/agent.py`
3. Run the extraction

### Modifying Output Schema

Edit the Pydantic models in `agents/agent.py`:

```python
class Diagnosis(BaseModel):
    term: str
    context: str
    temporal: str
    # Add new fields here
    severity: str
    icd_code: str
```

### Adjusting AI Parameters

Modify the `GenerateContentConfig` in `agents/agent.py`:

```python
config=types.GenerateContentConfig(
    response_mime_type="application/json",
    response_schema=DiagnosisList,
    temperature=0.1,  # Lower = more deterministic
    max_output_tokens=2048
)
```

## Dependencies

- **google-genai**: Google Gemini AI API client
- **pydantic**: Data validation and settings management
- **python-dotenv**: Environment variable management

## Troubleshooting

**API Key Issues:**
- Ensure `.env` file exists with valid `GEMINI_API_KEY`
- Check API key permissions and quotas

**Import Errors:**
- Run `uv sync` to install dependencies
- Verify Python version compatibility (3.12+)

**Text Processing:**
- Ensure input files are UTF-8 encoded
- Check file paths in the agent configuration

## License

This is a demonstration project for educational purposes.
