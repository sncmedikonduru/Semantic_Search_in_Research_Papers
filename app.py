from huggingface_hub import hf_hub_download
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import gradio as gr

# Ensure `torch` is imported and available
if not torch.cuda.is_available():
    print("CUDA is not available. Running on CPU.")

# Repository details
repo_id = "sncmedikonduru/Research"  # Your Hugging Face model repository
filename = "research_papers.parquet"  # Name of the .parquet file in the repository

# Download the .parquet file
print(f"Downloading {filename} from {repo_id}...")
try:
    parquet_path = hf_hub_download(repo_id=repo_id, filename=filename, force_download=True)
    print(f"File successfully downloaded to: {parquet_path}")
except Exception as e:
    raise FileNotFoundError(f"Failed to download {filename} from {repo_id}: {e}")

# Load the dataset
print(f"Loading dataset from {parquet_path}...")
data = pd.read_parquet(parquet_path)
print(f"Loaded {len(data)} records from the dataset.")

# Extract embeddings
embeddings = np.array(data["embedding"].tolist())
print(f"Extracted {embeddings.shape[0]} embeddings with dimension {embeddings.shape[1]}.")

# Model and tokenizer
model_checkpoint = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModel.from_pretrained(model_checkpoint)

# Define CLS pooling for query embeddings
def cls_pooling(model_output):
    """Extract CLS token embeddings from model output."""
    return model_output.last_hidden_state[:, 0]

# Function to compute embeddings for a query
def get_embeddings(query_list):
    """Compute embeddings for the input query list."""
    encoded_input = tokenizer(query_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        model_output = model(**encoded_input)
    return cls_pooling(model_output).cpu().detach().numpy()

# Semantic search function
def semantic_search(query, top_k=5):
    """Perform semantic search on the dataset."""
    # Compute the query embedding
    query_embedding = get_embeddings([query]).astype(np.float32)

    # Compute cosine similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]
    top_indices = similarities.argsort()[-top_k:][::-1]

    # Retrieve and format the top-k results
    results = []
    for idx in top_indices:
        results.append({
            "Title": data.iloc[idx]["title"],
            "Abstract": data.iloc[idx]["abstract"],
            "Similarity Score": round(similarities[idx], 4),
        })
    return results

# Gradio function for query input
def gradio_search(query):
    """Format search results for display in Gradio."""
    results = semantic_search(query)
    formatted_results = "\n\n".join([
        f"**Title:** {res['Title']}\n**Abstract:** {res['Abstract']}\n**Similarity Score:** {res['Similarity Score']}"
        for res in results
    ])
    return formatted_results

# Example queries
examples = [
    "What are the latest advancements in deep learning?",
    "Applications of machine learning in healthcare.",
    "Improving natural language processing models.",
]

# Gradio Interface
iface = gr.Interface(
    fn=gradio_search,
    inputs=gr.Textbox(lines=2, placeholder="Enter your query here..."),
    outputs="markdown",
    title="Semantic Search for Research Papers",
    description="Enter a query to find the most relevant research papers based on semantic similarity.",
    examples=examples,
)

# Launch the Gradio interface
iface.launch(share=True)
