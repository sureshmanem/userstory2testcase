# User Story to Manual Test Case Generator

A Python application that converts user stories to comprehensive manual test cases using Azure OpenAI (GPT models) with optional context retrieval from a local vector store. Features both CLI and Streamlit UI interfaces.

## Features

- **Dual Interface**: Command-line runner and interactive Streamlit web UI
- **Azure OpenAI Integration**: Uses official OpenAI Python SDK with AzureOpenAI client
- **Context Retrieval**: Build local vector stores from additional context and retrieve relevant information per user story
- **Automatic Fallback**: If HuggingFace embeddings fail (network/SSL/proxy issues), automatically falls back to offline TF-IDF
- **Comprehensive Output**: Generates manual tests with detailed metrics (tokens, latency, effort saved)
- **Flexible Configuration**: Properties files for prompts and I/O, environment variables for secrets

## Requirements

- Windows (optimized for Windows, but cross-platform compatible)
- Python 3.8 or higher
- Azure OpenAI account with a deployed GPT model

## Installation

### 1. Clone or Download

Download this repository to your local machine.

### 2. Create Virtual Environment (Windows PowerShell)

```powershell
# Create virtual environment
python -m venv .venv

# Activate virtual environment
.venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
```

### 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 4. Configure Azure OpenAI Credentials

Create a `.env` file in the project root (copy from `.env.example`):

```powershell
copy .env.example .env
```

Edit `.env` with your Azure OpenAI credentials:

```env
# Required
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=your_deployment_name

# Optional
AZURE_OPENAI_MODEL_NAME=gpt-35-turbo
```

**Security Note**: Never commit the `.env` file to version control. It's already included in `.gitignore`.

## Configuration Files

### Config/Config.properties

Controls LLM settings and embedding model configuration:

```ini
[AdvancedConfigurations]
embedding_model_name=thenlper/gte-small
embedding_model_path=Data/ExternalEmbeddingModel
external_model_threshold=0.156
default_model_threshold=0.50

[LLM]
LLM_Family=GPT
TEMPERATURE=0.05
```

### Config/ConfigGPT.properties

Contains all prompt templates for manual test generation. Modify these to customize the LLM's behavior.

### Config/ConfigIO.properties

Specifies input/output paths for CLI mode:

```ini
[Input]
additional_context_path=Data/Input/additionalContext.csv
input_file_path=Data/Input/user_stories.csv

[Output]
retrieval_context=Data/RetrievalContext
output_file_path=Data/Output
num_context_retrieve=3
manual_test_type=Both
```

## Usage

### Streamlit UI (Recommended)

Launch the web interface:

```powershell
streamlit run streamlit_app.py
```

Then:
1. Upload your user stories file (CSV or Excel)
2. Optionally upload additional context file
3. Configure generation options in the sidebar
4. Click "Generate Manual Test Cases"
5. View results and download as CSV

**Expected User Stories Format:**
- `Description`: User story description
- `AcceptanceCriteria`: Acceptance criteria
- `ID` / `FormattedID` / `UserStoryID`: (Optional) Unique identifier

### CLI Mode

1. Place your input files in `Data/Input/`:
   - `user_stories.csv` - User stories
   - `additionalContext.csv` - (Optional) Additional context

2. Run the CLI:

```powershell
python USManualTest.py
```

3. Follow the prompts to:
   - Use additional context (Yes/No)
   - Enable additional intelligence (Yes/No)
   - Generate additional acceptance criteria (Yes/No)

4. Output files are saved to `Data/Output/` with timestamps.

## Output Format

Generated CSV files contain:

| Column | Description |
|--------|-------------|
| UserStoryID | Unique identifier for the user story |
| TestType | Both, Positive, or Negative |
| UsedContext | Whether context was used (Yes/No) |
| UserStory | Original user story text |
| Context | Retrieved context (if any) |
| ManualTest | Generated manual test cases |
| LLMCalls | Number of LLM API calls |
| InputTokens | Total input tokens |
| OutputTokens | Total output tokens |
| TotalTokens | Total tokens used |
| LatencySeconds | Total generation time |
| TokensPerSecond | Throughput metric |
| ResponseModel | Model used for response |
| OutputWords | Word count of output |
| OutputChars | Character count of output |
| EstimatedEffortSavedMinutes | Estimated time saved (words/40) |
| PromptSentToLLM | Exact prompt sent to LLM |

## Context Retrieval

The system can use additional context to improve test generation:

1. **Vector Store Creation**: Converts context CSV into a local vector database
2. **Embedding Strategy**: 
   - Primary: HuggingFace embeddings (`thenlper/gte-small`)
   - Fallback: TF-IDF (offline, no network required)
3. **Retrieval**: For each user story, retrieves top-k most relevant context chunks
4. **Threshold Filtering**: Only uses context with similarity scores below threshold

### Corporate Environment Note

In corporate environments with strict network policies, HuggingFace model downloads may fail due to:
- Firewall/proxy restrictions
- SSL certificate validation issues
- Network timeouts

The system automatically detects these failures and falls back to TF-IDF, ensuring the pipeline continues working without interruption.

## Directory Structure

```
userstory2testcase/
├── Config/
│   ├── Config.properties          # LLM and embedding settings
│   ├── ConfigGPT.properties       # Prompt templates
│   └── ConfigIO.properties        # I/O paths for CLI
├── Data/
│   ├── Input/                     # Input CSV/Excel files
│   ├── Output/                    # Generated test cases
│   ├── RetrievalContext/          # Retrieved context logs
│   ├── SavedContexts/
│   │   ├── EmbedDataMTC/         # CLI vector store
│   │   └── EmbedDataMTC_UI/      # UI vector store
│   └── ExternalEmbeddingModel/   # HuggingFace model cache
├── us_to_mtc_file/
│   ├── __init__.py
│   ├── ChromaDBConnector.py      # Vector store (HF + TF-IDF)
│   ├── GenerateManualTest.py     # CLI processor
│   ├── GenerateManualTestResults.py  # Prompt builder
│   ├── ModelManualTestLLM.py     # Azure OpenAI client
│   └── ui_runner.py              # UI backend logic
├── tools/                         # Optional tools
├── .env                           # Environment variables (DO NOT COMMIT)
├── .env.example                   # Environment template
├── .gitignore
├── README.md
├── requirements.txt
├── streamlit_app.py              # Streamlit UI
└── USManualTest.py               # CLI entry point
```

## Troubleshooting

### Azure OpenAI Connection Fails

**Error**: Missing required environment variables

**Solution**: 
1. Ensure `.env` file exists in project root
2. Verify all required variables are set:
   - `AZURE_OPENAI_API_KEY`
   - `AZURE_OPENAI_ENDPOINT`
   - `AZURE_OPENAI_API_VERSION`
   - `AZURE_OPENAI_DEPLOYMENT_NAME`
3. Check that endpoint URL includes `https://` and trailing `/`

### HuggingFace Model Download Fails

**Error**: SSL, timeout, or connection errors when loading embeddings

**Solution**: This is expected in corporate environments. The system automatically falls back to TF-IDF. No action needed.

### CSV Encoding Issues

**Error**: UnicodeDecodeError when loading CSV files

**Solution**: The system tries multiple encodings (UTF-8, Latin-1). If issues persist, save your CSV as "UTF-8 with BOM" in Excel or your editor.

### Missing Columns in Input File

**Error**: KeyError for 'Description' or 'AcceptanceCriteria'

**Solution**: Ensure your user stories file contains these exact column names (case-sensitive):
- `Description`
- `AcceptanceCriteria`
- Optional: `ID`, `FormattedID`, or `UserStoryID`

## Development

### Project Structure

- **ModelManualTestLLM.py**: Azure OpenAI client wrapper with metrics tracking
- **ChromaDBConnector.py**: Local vector store with HF/TF-IDF fallback
- **GenerateManualTestResults.py**: Prompt template builder and LLM caller
- **GenerateManualTest.py**: CLI batch processor
- **ui_runner.py**: UI-focused batch processor with progress callbacks
- **USManualTest.py**: CLI entry point
- **streamlit_app.py**: Web UI

### Key Design Decisions

1. **No ChromaDB Dependency**: Uses simple file-based storage (JSONL, numpy, joblib)
2. **Automatic Fallback**: Graceful degradation from HF to TF-IDF
3. **Separation of Concerns**: CLI and UI logic separated for maintainability
4. **Environment-First Secrets**: All secrets from environment variables, never hardcoded

## License

[Your License Here]

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review configuration files for correctness
3. Ensure Azure OpenAI credentials are valid
4. Check Python and dependency versions

## Version

1.0.0
# userstory2testcase
