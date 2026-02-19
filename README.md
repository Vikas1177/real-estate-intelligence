# Real Estate Intelligence Q&A

[![Ask DeepWiki](https://devin.ai/assets/askdeepwiki.png)](https://deepwiki.com/Vikas1177/real-estate-intelligence)

This repository provides a high-performance, two-stage semantic search system designed to answer questions from a corpus of real estate documents in PDF format. It uses state-of-the-art embedding and cross-encoder models to deliver accurate results with low latency.

The system features an ingestion pipeline to process and index PDF documents and a FastAPI-based query service for real-time searching.

## Features

-   **Intelligent PDF Processing**: Extracts both text and structured tables from PDFs using PyMuPDF.
-   **OCR Fallback**: Automatically uses Tesseract OCR for pages with low text content (e.g., image-based pages) to ensure comprehensive data extraction.
-   **Advanced Retrieval-Reranking Pipeline**:
    -   **Stage 1 (Retrieval)**: Fast and efficient candidate retrieval using a `BAAI/bge-base-en-v1.5` embedding model and a FAISS (HNSW) vector index.
    -   **Stage 2 (Reranking)**: High-precision scoring of initial candidates using a `cross-encoder/ms-marco-MiniLM-L-6-v2` model to improve relevance.
-   **API Service**: A `FastAPI` server provides a simple `/search` endpoint for easy integration.
-   **Dockerized**: Comes with a `Dockerfile` for easy setup and deployment.
-   **Evaluation Suite**: Includes a script (`evaluate.py`) to measure the system's accuracy (Top-1, Top-3) and average latency against a predefined test set.

## Architecture

The system is composed of two main components: the Ingestion Pipeline and the Query Service.

### 1. Ingestion Pipeline (`index_gen.py`)

This script processes PDF documents and builds the necessary index files for searching.

1.  **Text Extraction**: PDF files are processed page by page. Text and tables are extracted. If a page contains fewer than 35 words, OCR is performed to capture text from images.
2.  **Chunking**: The extracted text from each page is split into smaller, overlapping chunks of approximately 300 words.
3.  **Embedding**: Each text chunk is converted into a vector embedding using the `BAAI/bge-base-en-v1.5` model.
4.  **Indexing**: The generated embeddings are stored in a FAISS `IndexHNSWFlat` index for fast and scalable similarity searches.
5.  **Metadata Storage**: Corresponding text, page numbers, and PDF source information for each chunk are saved in a `metadata.jsonl` file.

### 2. Query Service (`main.py`)

This FastAPI application serves search requests.

1.  **Receive Query**: The service accepts a POST request to `/search` with a user's query.
2.  **Embed Query**: The incoming query string is converted into a vector embedding using the same `bge-base-en-v1.5` model.
3.  **Initial Retrieval**: The FAISS index is searched to retrieve the top 10 most similar text chunks.
4.  **Reranking**: The cross-encoder model then re-scores these 10 candidates by directly comparing the query text with the chunk text. This significantly improves the final ranking accuracy.
5.  **Return Results**: The top-k results after reranking are returned as a JSON response, including the text snippet, source page number, PDF name, and relevance score.

## Setup and Usage

### Prerequisites

-   Python 3.10+
-   Tesseract OCR
    -   **On Debian/Ubuntu**: `sudo apt-get install tesseract-ocr`
    -   **On macOS**: `brew install tesseract`
    -   **On Windows**: Download from the [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki) page.

### 1. Installation

Clone the repository and install the required Python packages.

```bash
git clone https://github.com/vikas1177/real-estate-intelligence.git
cd real-estate-intelligence
pip install -r requirements.txt
```

### 2. Ingestion

Before you can run queries, you must first process your PDF documents and create the search index.

1.  Create a directory and place your PDF files inside it (e.g., `data/pdfs/`).
2.  Run the ingestion script. This will create an `index_data` directory containing the FAISS index and metadata files.

```bash
# Example: Process PDFs located in a 'docs' directory
python index_gen.py --pdf_dir docs --out_dir ./index_data
```

### 3. Run the API Service

Start the FastAPI server using Uvicorn.

```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`.

### 4. Query the System

You can send POST requests to the `/search` endpoint to get answers.

**Example using `curl`:**

```bash
curl -X POST "http://localhost:8000/search" \
     -H "Content-Type: application/json" \
     -d '{"query": "What LEED certification does Max Towers have?", "k": 3}'
```

**Example Response:**

```json
{
  "results": [
    {
      "score": 9.87321,
      "text": "Max Towers is an IGBC Platinum Certified building under the Health and Well-being rating system... It is a LEED Platinum certified development.",
      "pdf_name": "Max-Estates-Brochure.pdf",
      "page": 11
    },
    ...
  ],
  "latency_ms": 150.75
}
```

## Evaluation

The repository includes a script to evaluate the system's performance on a predefined set of questions.

1.  Make sure the API service is running (see Step 3 above).
2.  Run the evaluation script:

```bash
python evaluate.py
```

The script will query the API for each question in its test set and print the retrieved results, latency, and final accuracy metrics.

**Example Output:**

```
--------------------------------------------------
QUERY: 'What LEED certification does Max Towers have?'
EXPECTED PAGE: 11
LATENCY: 145.31 ms

RETRIEVED RESULTS:
  1. [Page 11] (Score: 9.8732) [CORRECT]
     Text: Max Towers is an IGBC Platinum Certified building under the Health and Well-being rating system... It is a LEED Platinum certified development.
...

FINAL SUCCESS METRICS
Average Query Latency: 152.41 ms
Top-1 Accuracy:        90.6%
Top-3 Accuracy:        96.9%
```

## Docker

You can also build and run the application using Docker.

1.  **Build the Docker Image:**
    Make sure you have already run the ingestion step to create the `./index_data` directory, as this will be copied into the container.

    ```bash
    docker build -t real-estate-intelligence .
    ```

2.  **Run the Docker Container:**
    This command runs the container and maps port 8000 from the container to your local machine.

    ```bash
    docker run -p 8000:8000 real-estate-intelligence
    ```

    The service will be accessible at `http://localhost:8000`.
