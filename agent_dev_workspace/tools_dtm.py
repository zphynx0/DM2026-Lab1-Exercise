from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool
from sklearn.feature_extraction.text import CountVectorizer

MAX_TOP_TERMS = 50


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
        top_n: Optional[int] = 20
    ) -> Dict[str, Any]:
        """Aggregates term frequencies across all documents in the Document-Term Matrix.
 
        Sums columns of session.feature_matrix across all documents to compute total frequency per term.
        Only the top_n most frequent terms are returned (capped at 50 to keep the response small);
        the full ranked table is stored for visualize_result_tool.
 
        Args:
            top_n: Number of top-frequent terms to include in the returned summary dict (default: 20, max: 50).
 
        Returns:
            Dict containing result_id, total_terms, and top_terms mapping terms to their frequency counts.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix in session. Run build_dtm_tool first."}
 
        # Resolve and clamp top_n so the returned dict is always small
        limit = 20 if top_n is None else int(top_n)
        if limit < 1:
            return {"status": "error", "message": "top_n must be at least 1."}
        clamped = limit > MAX_TOP_TERMS
        limit = min(limit, MAX_TOP_TERMS)
 
        # Sum occurrences across all documents for each term column
        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        terms = session.feature_names
 
        # Stable descending sort (ties keep vocabulary order)
        order = np.argsort(-counts, kind="stable")
 
        reported_tf = {terms[i]: int(counts[i]) for i in order[:limit]}
 
        # Full ranked report for visualize_result_tool (first col: term, second col: count); never sent to the LLM
        df_report = pd.DataFrame({
            "term": [terms[i] for i in order],
            "frequency": counts[order].astype(int),
        })
 
        result_id = session.next_result_id("tf")
        summary = {
            "result_id": result_id,
            "total_terms": len(terms),
            "top_terms": reported_tf
        }
        if clamped:
            summary["note"] = f"top_n was capped at {MAX_TOP_TERMS} to keep the response small."
 
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
