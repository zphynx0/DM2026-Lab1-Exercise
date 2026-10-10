from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer


def make_tools(session):
    @tool
    def build_dtm_tool(
        max_features: Optional[int] = None,
        ngram_range: List[int] = [1, 1],
        lowercase: bool = True,
    ) -> Dict[str, Any]:
        """Builds a document-term matrix (DTM) from the dataset text.

        Constructs a sparse document-term matrix using scikit-learn's CountVectorizer,
        updating the session state with the feature matrix, feature names, and fitted vectorizer.
        Computes matrix sparsity metrics.

        Args:
            max_features: Maximum number of terms to include in the vocabulary ordered by term frequency.
            ngram_range: Lower and upper boundary of the range of n-values for different n-grams to be extracted, e.g. [1, 1] for unigrams.
            lowercase: Whether to convert all characters to lowercase before tokenizing.

        Returns:
            Dict containing summary statistics of the built DTM (n_documents, n_terms, sparsity_pct, etc.).
        """
        if session.dataframe is None or "text" not in session.dataframe.columns:
            return {"status": "error", "message": "No dataset loaded in session. Load a dataset first."}

        # Handle missing text gracefully
        texts = session.dataframe["text"].fillna("").astype(str)

        # Convert ngram_range list parameter to tuple for CountVectorizer
        ngram_tuple = (ngram_range[0], ngram_range[1]) if len(ngram_range) >= 2 else (1, 1)

        vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_tuple,
            lowercase=lowercase
        )

        feature_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()

        session.feature_matrix = feature_matrix
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        # Set category labels on session if present
        if "category_name" in session.dataframe.columns:
            session.set_labels(session.dataframe["category_name"].values)
        elif "label" in session.dataframe.columns:
            session.set_labels(session.dataframe["label"].values)

        n_rows, n_cols = feature_matrix.shape
        non_zero = int(feature_matrix.nnz)
        total_elements = n_rows * n_cols
        sparsity_pct = 100.0 * (1.0 - non_zero / total_elements) if total_elements > 0 else 0.0

        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": n_rows,
            "n_terms": n_cols,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": round(sparsity_pct, 4),
            "vocabulary_sample": feature_names[:10]
        }

        session.store_result("build_dtm_tool", {"max_features": max_features, "ngram_range": ngram_range, "lowercase": lowercase}, summary)

        return summary

    @tool
    def term_frequency_tool(
        top_n: Optional[int] = None
    ) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the Document-Term Matrix.

        Sums columns of session.feature_matrix across all documents to compute total frequency per term.

        Args:
            top_n: Optional limit on the number of top-frequent terms to return.

        Returns:
            Dict containing total_terms, term_frequencies (term -> total_count), and top_terms.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix in session. Run build_dtm_tool first."}

        # Sum occurrences across all documents for each term column
        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten().tolist()
        terms = session.feature_names

        # Create dictionary mapping term -> total count
        tf_dict = {term: int(count) for term, count in zip(terms, counts)}

        # Sort descending by frequency
        sorted_tf = dict(sorted(tf_dict.items(), key=lambda item: item[1], reverse=True))

        if top_n is not None:
            items = list(sorted_tf.items())[:top_n]
            reported_tf = dict(items)
        else:
            reported_tf = sorted_tf

        # Format full report DataFrame for visualize_result_tool compatibility
        df_report = pd.DataFrame(list(sorted_tf.items()), columns=["term", "frequency"])

        result_id = session.next_result_id("tf")
        summary = {
            "result_id": result_id,
            "total_terms": len(terms),
            "term_frequencies": tf_dict,
            "top_terms": reported_tf
        }

        session.store_result("term_frequency_tool", {"top_n": top_n}, summary, full_report=df_report)

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20
    ) -> Dict[str, Any]:
        """Renders a heatmap of a raw positional slice of the document-term matrix.

        Slices session.feature_matrix[0:n_documents, 0:n_terms] in existing order without sorting.
        Plots an annotated count heatmap onto session.pending_figure and returns the matrix.

        Args:
            n_terms: Number of terms (columns) to include from session.feature_names (default: 20).
            n_documents: Number of documents (rows) to include from session.dataframe (default: 20).

        Returns:
            Dict containing result_id, n_terms, n_documents, terms, documents, and matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix in session. Run build_dtm_tool first."}

        m_docs, n_cols = session.feature_matrix.shape

        if m_docs == 0 or n_cols == 0:
            return {"status": "error", "message": "Document-term matrix is empty."}

        slice_terms = min(n_terms, n_cols)
        slice_docs = min(n_documents, m_docs)

        selected_terms = session.feature_names[0:slice_terms]
        doc_indices = list(range(slice_docs))

        sub_matrix = session.feature_matrix[0:slice_docs, 0:slice_terms].toarray()
        matrix_list = sub_matrix.tolist()

        # Build heatmap plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(
            sub_matrix,
            xticklabels=selected_terms,
            yticklabels=doc_indices,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax,
        )
        ax.set_title(f"DTM Heatmap ({slice_docs} Docs x {slice_terms} Terms)")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Document Index")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_terms": slice_terms,
            "n_documents": slice_docs,
            "terms": selected_terms,
            "documents": doc_indices,
            "matrix": matrix_list,
        }

        df_report = pd.DataFrame(sub_matrix, index=doc_indices, columns=selected_terms)

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
            full_report=df_report,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
