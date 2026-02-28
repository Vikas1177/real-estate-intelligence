# Real Estate Intelligence Q&A

This repository provides a high-performance, two-stage semantic search system designed to answer questions from a corpus of real estate documents in PDF format. It uses state-of-the-art embedding and cross-encoder models to deliver accurate results with low latency.
>**Note on System Design & Scalability:** For a detailed breakdown of our technical decisions, system limits, and bottleneck analysis, please read the #[Architecture & System Design](ARCHITECTURE.md).
>**Video demo**[DEMO](https://drive.google.com/file/d/15_XwfioKKHdqjkKyz1yMMa2TXko3d6gm/view?usp=sharing)

```
graph TD
    subgraph Data Sources
        Docs[PDF Documents]
    end

    subgraph Offline Ingestion Pipeline
        direction TB
        Proc(<b>processing.py</b><br/>PyMuPDF Extraction & Tesseract OCR)
        Chunk(Sentence-Aware Chunking<br/>400 words / 80 overlap)
        Embed1(<b>index_gen.py</b><br/>Embedding Generation<br/><i>BAAI/bge-large-en-v1.5</i>)
        
        Docs --> Proc
        Proc --> Chunk
        Chunk --> Embed1
    end

    subgraph Vector Database
        DB[(FAISS HNSW Index <br/>+ Metadata.jsonl)]
        Embed1 -->|Indexes & Saves| DB
    end

    subgraph Online Retrieval Engine
        direction TB
        API(<b>main.py</b><br/>FastAPI Endpoint)
        Embed2(Query Embedding<br/><i>BAAI/bge-large-en-v1.5</i>)
        FAISS(FAISS ANN Search<br/>Retrieves Top-80 Candidates)
        Rerank(Cross-Encoder Reranking<br/><i>ms-marco-MiniLM-L-6-v2</i>)
        
        API --> Embed2
        Embed2 --> FAISS
        DB -.->|Loaded in RAM| FAISS
        FAISS -->|Top 80| Rerank
        Rerank -->|Scores Top 25| Top3[Returns Top 3 Results]
    end

    subgraph Clients & Evaluation
        User((User / Frontend))
        Eval(<b>evaluate.py</b><br/>Test Queries & Metrics)
        
        User <-->|JSON Requests / Responses| API
        Eval -.->|Automated Tests <br/> Recall@K & Latency| API
    end

 ```   
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

**Stage-wise Latency Breakdown**
FOR MAX HOUSE OKHLA [PDF](https://maxestates.in/wp-content/themes/max-estate/html/pdf/max-house-brochure.pdf)
```
Total Time:   28.59s
  - Extraction: 8.02s
  - Chunking:   0.00s
  - Metadata:   0.00s
  - Embedding:  20.56s
  - Indexing:   0.01s
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
curl -X 'POST' \
  'http://127.0.0.1:8000/search' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What is the green rating of Max House?",
  "k": 3
}'
```

**Example Response:**

```json
{
  "results": [
    {
      "score": 4.953125,
      "text": "Open for business; In the heart of New Delhi Originally the headquarters of the $3B Max Group, Max House is located at the epicenter of the Secondary Business District of Delhi. Offering ~105,000 sq. ft. of prime real estate spread across 10 floors, Max House is poised to be the new business address in Delhi NCR. Designed to address the future of work while considering human capital to be an occupier's most important resource, Max House blends thoughtful design and superior hospitality in order to nurture a more productive, healthier and happier community. 3Ma\u0003.I\u0003K\\[ 1,05,425 sq.ft. Super Built Up Area Building Height 40 m Total Number of Floors 1 Basement Parking 1 Stilt Parking 2 Podium Level Parking 8 Tenant Floors Terrace Typical Floor Plate Size 13,000 sq. ft. Green Rating LEED Gold F...",
      "pdf_name": "max-house-brochure.pdf",
      "page": 2
    },
    {
      "score": -0.806640625,
      "text": "Sustainability: LEED Gold Certified “Ultimately, we are responsible for building the future we want.” Max House is designed to be LEED Gold certified. Max House is a thought leader in sustainability and aims to minimise its ecological footprint. To do so is important to us because we feel a certain responsibility towards our planet, and we invite you to share our enthusiasm for the same. The LEED Gold certification is a validation of our efforts and helps cement our belief that ecology, biophilia, commerce and real estate can co-exist at a single, iconic address. \u0014\u0013",
      "pdf_name": "max-house-brochure.pdf",
      "page": 10
    },
    {
      "score": -1.99609375,
      "text": "Simply Superior Materials & Construction The materials used to build each space have been carefully chosen to maintain a sense of luxury and balancing it with our high sustainability design standards. From unique glass bricks to terracotta brick tiles, everything at Max House has been hand-picked to ensure a sense of luxury, comfort and longevity. The common areas of Max House are well appointed with: A combination of brick and marble which exude a sense of welcome and warmth Double glazed windows to lower operating costs while allowing light transmission Wooden wall finish & panelling High ceilings with a height of 3.75 meters Efficient floor plates with a clean, efficient rectangular design Plants and nature wherever you look Lobby of Max House \u001a \u0017",
      "pdf_name": "max-house-brochure.pdf",
      "page": 7
    }
  ],
  "total_ms": 182.08,
  "embed_ms": 92.98,
  "retrieve_ms": 0.12,
  "rerank_ms": 88.53,
  "rerank_token_ms": 10.48,
  "rerank_forward_ms": 10.32
}
```

## Evaluation

The repository includes a script to evaluate the system's performance on a predefined set of questions.

1.  Make sure the API service is running (see Step 3 above).
2.  Run the evaluation script:

```bash
python evaluate.py
```


**FINAL SUCCESS METRICS**

```
Average Query Latency: 1002.25 ms
P95 Latency:           1506.51 ms
Top-1 Accuracy:        84.62%
Top-3 Accuracy:        92.31%
MRR: 0.8846
nDCG@1: 0.8462
nDCG@3: 0.8947
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
