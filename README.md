
# Semantic Search in ML Research Papers

This project implements a semantic search application using a subset of Machine Learning papers from the ArXiv dataset, specifically filtered and pre-processed for semantic analysis. The dataset is sourced from Hugging Face's datasets library under the repository 'CShorten/ML-ArXiv-Papers'.

## Dataset Description

The dataset includes around 100,000 ML papers tagged with 'cs.LG' from the full ArXiv dataset which contains approximately 2 million papers. For this project, only the first 50,000 papers are used after combining their titles and abstracts into a single text field. This preprocessing step is crucial for improving the performance of semantic search by broadening the text context available for analysis.

## Features

- **Semantic Search**: Allows users to input a query related to machine learning and find the most relevant papers based on semantic content.
- **Efficient Preprocessing**: Combines titles and abstracts of papers to enhance the quality of text data fed into the model.
- **Gradio Interface**: An easy-to-use web interface that allows users to perform searches and view results, including paper titles, abstracts, and similarity scores.

## How It Works

1. **Dataset Loading and Preprocessing**: The ArXiv dataset is loaded and preprocessed using the following steps:
   - Titles and abstracts are combined to enrich the text data.
   - The dataset is limited to the first 50,000 samples to maintain manageability and performance.

2. **Semantic Analysis**:
   - The combined text data is used to generate embeddings that capture the semantic meaning of the text.
   - A search function calculates the cosine similarity between the query's embedding and the embeddings of the dataset's entries to find the most relevant papers.

## Setup

To set up and run this project:

1. Install the required libraries:
   \```bash
   pip install datasets transformers torch gradio pandas numpy scikit-learn
   \```

2. Download and prepare the dataset:
   \```python
   from datasets import load_dataset

   # Load the ArXiv dataset
   arxiv_dataset = load_dataset('CShorten/ML-ArXiv-Papers', trust_remote_code=True)

   # Preprocess the dataset
   arxiv_dataset = arxiv_dataset['train'].map(combine_title_abstract)
   arxiv_dataset = arxiv_dataset.select(range(50000))
   \```

3. Run the application:
   - Start the Gradio interface and input queries to perform semantic searches.

## Example Queries

- "What are the latest advancements in deep learning?"
- "Applications of machine learning in healthcare."
- "Improving natural language processing models."

## License

This project is open-sourced under the MIT license.
