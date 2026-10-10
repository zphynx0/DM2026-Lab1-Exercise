from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def feature_correlation_matrix_tool() -> Dict[str, Any]:
        """Computes a Pearson correlation matrix and heatmap over the top-20 terms by variance.

        Operates on the global document-term matrix (session.feature_matrix / session.feature_names).
        Filters to the top 20 terms with highest variance across documents, computes their pairwise
        Pearson correlation matrix, plots an annotated heatmap onto session.pending_figure, and
        returns the correlation matrix and term list in the summary.

        Returns:
            Dict containing result_id, terms, and correlation_matrix (nested list of floats).
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {
                "status": "error",
                "message": "No document-term matrix in session. Run build_dtm_tool first."
            }

        dense = session.feature_matrix.toarray()
        n_docs, n_terms = dense.shape

        if n_terms == 0:
            return {
                "status": "error",
                "message": "Document-term matrix has no terms."
            }

        # Step 1: Compute variance for each term across all documents
        variances = np.var(dense, axis=0)

        # Step 2: Pick top 20 terms by variance (or all if < 20 terms)
        top_k = min(20, n_terms)
        top_indices = np.argsort(variances)[::-1][:top_k]

        top_terms = [session.feature_names[i] for i in top_indices]
        sub_matrix = dense[:, top_indices]

        # Step 3: Compute Pearson correlation matrix across top terms' columns
        # np.corrcoef expects variables as rows if rowvar=True, so rowvar=False treats columns as variables
        corr_matrix = np.corrcoef(sub_matrix, rowvar=False)

        # Handle edge cases (e.g., 1 term or zero variance resulting in NaNs)
        if top_k == 1:
            corr_matrix = np.array([[1.0]])
        else:
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

        # Step 4: Build heatmap plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            ax=ax,
            cbar=True,
        )
        ax.set_title("Top Feature Correlation Matrix")
        plt.tight_layout()

        session.pending_figure = fig

        # Step 5: Store result and return summary
        result_id = session.next_result_id("feature_corr")
        corr_list = [[round(float(val), 4) for val in row] for row in corr_matrix]

        summary = {
            "result_id": result_id,
            "terms": top_terms,
            "correlation_matrix": corr_list,
        }

        # Prepare DataFrame for store_result full_report
        df_report = pd.DataFrame(corr_list, index=top_terms, columns=top_terms)

        session.store_result(
            "feature_correlation_matrix_tool",
            {},
            summary,
            full_report=df_report,
        )

        return summary

    @tool
    def cosine_similarity_tool(
        doc_index1: int,
        doc_index2: int
    ) -> Dict[str, Any]:
        """Computes cosine similarity between two documents in the Document-Term Matrix.

        Retrieves row vectors for doc_index1 and doc_index2 from session.feature_matrix and computes
        their cosine similarity (dot product divided by product of norms). Zero vectors return 0.0.

        Args:
            doc_index1: Row index of the first document (0-based integer).
            doc_index2: Row index of the second document (0-based integer).

        Returns:
            Dict containing result_id, doc_index1, doc_index2, and cosine_similarity (float).
        """
        if session.feature_matrix is None:
            return {
                "status": "error",
                "message": "No document-term matrix in session. Run build_dtm_tool first."
            }

        n_docs = session.feature_matrix.shape[0]

        if doc_index1 < 0 or doc_index1 >= n_docs:
            return {
                "status": "error",
                "message": f"doc_index1={doc_index1} out of bounds (matrix has {n_docs} documents)."
            }

        if doc_index2 < 0 or doc_index2 >= n_docs:
            return {
                "status": "error",
                "message": f"doc_index2={doc_index2} out of bounds (matrix has {n_docs} documents)."
            }

        u = session.feature_matrix[doc_index1].toarray().flatten()
        v = session.feature_matrix[doc_index2].toarray().flatten()

        norm_u = np.linalg.norm(u)
        norm_v = np.linalg.norm(v)

        if norm_u == 0.0 or norm_v == 0.0:
            sim = 0.0
        else:
            sim = float(np.dot(u, v) / (norm_u * norm_v))

        sim_rounded = round(sim, 4)

        result_id = session.next_result_id("cosine_sim")
        summary = {
            "result_id": result_id,
            "doc_index1": doc_index1,
            "doc_index2": doc_index2,
            "cosine_similarity": sim_rounded,
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc_index1": doc_index1, "doc_index2": doc_index2},
            summary,
        )

        return summary

    return [feature_correlation_matrix_tool, cosine_similarity_tool]
