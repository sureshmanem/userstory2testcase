"""
ChromaDBConnector.py
Implements local vector store with HuggingFace embeddings and TF-IDF fallback.
No chromadb dependency - uses simple file-based storage.
"""

import os
import json
import numpy as np
import configparser
from datetime import datetime
from typing import Tuple, Dict, Optional
from langchain_community.document_loaders import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
from scipy import sparse


class ChromaDBConnector:
    """
    Local vector store connector with automatic fallback to TF-IDF.
    Stores documents as JSONL and embeddings as numpy arrays or sparse matrices.
    """

    def __init__(self, persist_directory: str):
        """
        Initialize the connector with a persistence directory.

        Args:
            persist_directory: Path to store the vector database files.
        """
        self.persist_directory = persist_directory
        self.documents_file = os.path.join(persist_directory, "documents.jsonl")
        self.embeddings_file = os.path.join(persist_directory, "embeddings.npy")
        self.meta_file = os.path.join(persist_directory, "store_meta.json")
        self.tfidf_vectorizer_file = os.path.join(persist_directory, "tfidf_vectorizer.joblib")
        self.tfidf_matrix_file = os.path.join(persist_directory, "tfidf_matrix.npz")

        # Read configuration
        config = configparser.ConfigParser()
        config.read("Config/Config.properties")
        self.embedding_model_name = config.get("AdvancedConfigurations", "embedding_model_name")
        self.embedding_model_path = config.get("AdvancedConfigurations", "embedding_model_path")
        self.external_model_threshold = float(config.get("AdvancedConfigurations", "external_model_threshold"))
        self.default_model_threshold = float(config.get("AdvancedConfigurations", "default_model_threshold"))

    def vector_store(self, context_csv_path: str):
        """
        Build vector store from CSV file with automatic fallback to TF-IDF.

        Args:
            context_csv_path: Path to the CSV file containing context data.
        """
        # Create persist directory if it doesn't exist
        os.makedirs(self.persist_directory, exist_ok=True)

        # Load documents from CSV
        print(f"Loading documents from {context_csv_path}...")
        try:
            loader = CSVLoader(file_path=context_csv_path, encoding="utf-8")
            documents = loader.load()
        except UnicodeDecodeError:
            loader = CSVLoader(file_path=context_csv_path, encoding="latin-1")
            documents = loader.load()

        if not documents:
            raise ValueError(f"No documents loaded from {context_csv_path}")

        # Save documents as JSONL
        with open(self.documents_file, "w", encoding="utf-8") as f:
            for doc in documents:
                f.write(json.dumps({"page_content": doc.page_content}) + "\n")

        print(f"Saved {len(documents)} documents to {self.documents_file}")

        # Try HuggingFace embeddings first
        try:
            print(f"Attempting to load HuggingFace model: {self.embedding_model_name}...")
            embeddings_model = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                cache_folder=self.embedding_model_path
            )

            # Embed all documents
            texts = [doc.page_content for doc in documents]
            print("Generating embeddings...")
            embeddings = embeddings_model.embed_documents(texts)
            embeddings_array = np.array(embeddings)

            # Save embeddings
            np.save(self.embeddings_file, embeddings_array)

            # Save metadata
            meta = {
                "store_type": "hf_dense",
                "model_name": self.embedding_model_name,
                "num_documents": len(documents),
                "embedding_dim": embeddings_array.shape[1],
                "threshold": self.external_model_threshold
            }
            with open(self.meta_file, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"Successfully created HuggingFace dense vector store with {len(documents)} documents.")

        except Exception as e:
            print(f"HuggingFace embeddings failed: {str(e)}")
            print("Falling back to TF-IDF...")

            # Fallback to TF-IDF
            texts = [doc.page_content for doc in documents]
            vectorizer = TfidfVectorizer(stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Save TF-IDF artifacts
            joblib.dump(vectorizer, self.tfidf_vectorizer_file)
            sparse.save_npz(self.tfidf_matrix_file, tfidf_matrix)

            # Save metadata
            meta = {
                "store_type": "tfidf_sparse",
                "num_documents": len(documents),
                "vocabulary_size": len(vectorizer.vocabulary_),
                "threshold": self.default_model_threshold
            }
            with open(self.meta_file, "w") as f:
                json.dump(meta, f, indent=2)

            print(f"Successfully created TF-IDF sparse vector store with {len(documents)} documents.")

    def retrieval_context(self, query: str, k: int = 3) -> Tuple[str, Dict[float, str], float]:
        """
        Retrieve top-k relevant contexts for a query.

        Args:
            query: The query string.
            k: Number of top results to retrieve.

        Returns:
            Tuple of (combined_context_text, docs_with_similarity_score_dict, threshold)
            where docs_with_similarity_score_dict maps score->doc_text.
        """
        # Load metadata
        if not os.path.exists(self.meta_file):
            raise FileNotFoundError(f"Vector store not found in {self.persist_directory}")

        with open(self.meta_file, "r") as f:
            meta = json.load(f)

        store_type = meta["store_type"]
        threshold = meta["threshold"]

        # Load documents
        documents = []
        with open(self.documents_file, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc["page_content"])

        # Retrieve based on store type
        if store_type == "tfidf_sparse":
            scores = self._retrieve_tfidf(query, documents, k)
        elif store_type == "hf_dense":
            try:
                scores = self._retrieve_hf(query, documents, k)
            except Exception as e:
                print(f"HuggingFace retrieval failed: {str(e)}")
                # Check if TF-IDF artifacts exist
                if os.path.exists(self.tfidf_vectorizer_file):
                    print("Falling back to TF-IDF for retrieval...")
                    # Update meta to tfidf
                    meta["store_type"] = "tfidf_sparse"
                    meta["threshold"] = self.default_model_threshold
                    with open(self.meta_file, "w") as f:
                        json.dump(meta, f, indent=2)
                    threshold = meta["threshold"]
                    scores = self._retrieve_tfidf(query, documents, k)
                else:
                    raise RuntimeError("HuggingFace retrieval failed and no TF-IDF fallback available")
        else:
            raise ValueError(f"Unknown store type: {store_type}")

        # Build result dict with unique scores (add tiny offset to avoid collisions)
        docs_with_scores = {}
        for idx, score in scores:
            # Ensure uniqueness by adding tiny offset
            unique_score = score
            offset = 0
            while unique_score in docs_with_scores:
                offset += 1
                unique_score = score + (offset * 1e-12)
            docs_with_scores[unique_score] = documents[idx]

        # Combine context
        combined_context = "\n\n".join([documents[idx] for idx, _ in scores])

        # Save retrieved context to timestamped file
        config = configparser.ConfigParser()
        config.read("Config/ConfigIO.properties")
        retrieval_context_dir = config.get("Output", "retrieval_context")
        os.makedirs(retrieval_context_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        retrieval_file = os.path.join(retrieval_context_dir, f"retrieved_{timestamp}.txt")
        with open(retrieval_file, "a", encoding="utf-8") as f:
            f.write(f"Query: {query}\n")
            f.write(f"Retrieved at: {datetime.now().isoformat()}\n")
            f.write(f"Top {k} contexts:\n")
            f.write("=" * 80 + "\n")
            f.write(combined_context)
            f.write("\n" + "=" * 80 + "\n\n")

        return combined_context, docs_with_scores, threshold

    def _retrieve_tfidf(self, query: str, documents: list, k: int) -> list:
        """
        Retrieve using TF-IDF.

        Args:
            query: Query string.
            documents: List of document texts.
            k: Number of results.

        Returns:
            List of (doc_index, score) tuples sorted by score (lower is better).
        """
        # Load TF-IDF artifacts
        vectorizer = joblib.load(self.tfidf_vectorizer_file)
        tfidf_matrix = sparse.load_npz(self.tfidf_matrix_file)

        # Transform query
        query_vec = vectorizer.transform([query])

        # Compute cosine similarity
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

        # Convert to distance-like score (lower is better)
        scores = 1 - similarities

        # Get top-k indices
        top_indices = np.argsort(scores)[:k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]

    def _retrieve_hf(self, query: str, documents: list, k: int) -> list:
        """
        Retrieve using HuggingFace embeddings.

        Args:
            query: Query string.
            documents: List of document texts.
            k: Number of results.

        Returns:
            List of (doc_index, score) tuples sorted by score (lower is better).
        """
        # Load embeddings
        embeddings_array = np.load(self.embeddings_file)

        # Load embedding model
        embeddings_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name,
            cache_folder=self.embedding_model_path
        )

        # Embed query
        query_embedding = np.array(embeddings_model.embed_query(query)).reshape(1, -1)

        # Compute cosine similarity
        similarities = cosine_similarity(query_embedding, embeddings_array).flatten()

        # Convert to distance-like score (lower is better)
        scores = 1 - similarities

        # Get top-k indices
        top_indices = np.argsort(scores)[:k]

        return [(int(idx), float(scores[idx])) for idx in top_indices]
