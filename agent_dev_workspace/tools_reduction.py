from typing import Dict, Any, List, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.figure
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        n_components: int = 2,
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduces document-term matrix dimensions to lower-dimensional coordinates (PCA/t-SNE/UMAP).

        Fits requested dimensionality reduction method on session.feature_matrix.
        Plots a scatter plot of the reduced 2D coordinates colored by category onto session.pending_figure,
        and returns coordinates and summary stats.

        Args:
            method: Dimensionality reduction algorithm ("pca", "tsne", or "umap"). Default: "pca".
            n_components: Number of target dimensions (default: 2).
            perplexity: Perplexity parameter for t-SNE (default: 30.0).
            n_neighbors: Number of neighbors for UMAP (default: 15).
            random_state: Random state seed for reproducibility (default: 42).

        Returns:
            Dict containing result_id, method, n_components, coordinates, and explained_variance_ratio (for PCA).
        """
        if session.feature_matrix is None:
            return {
                "status": "error",
                "message": "No document-term matrix in session. Run build_dtm_tool first."
            }

        X = session.feature_matrix.toarray()
        n_samples = X.shape[0]

        if n_samples == 0:
            return {
                "status": "error",
                "message": "Document-term matrix has 0 documents."
            }

        # Extract category labels for coloring the scatter plot
        labels = None
        if session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].tolist()
            elif "category" in session.dataframe.columns:
                labels = session.dataframe["category"].tolist()

        if labels is None:
            labels = ["Unlabeled"] * n_samples

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=n_components, random_state=random_state)
            coords_arr = reducer.fit_transform(X)
            explained_variance_ratio = [
                round(float(val), 4) for val in reducer.explained_variance_ratio_
            ]
        elif method == "tsne":
            reducer = TSNE(
                n_components=n_components,
                perplexity=perplexity,
                random_state=random_state,
            )
            coords_arr = reducer.fit_transform(X)
        elif method == "umap":
            reducer = umap.UMAP(
                n_components=n_components,
                n_neighbors=n_neighbors,
                random_state=random_state,
            )
            coords_arr = reducer.fit_transform(X)
        else:
            return {
                "status": "error",
                "message": f"Unsupported method '{method}'. Supported methods: 'pca', 'tsne', 'umap'."
            }

        # Store coordinates in session.artifacts
        session.artifacts[f"reduced_coords_{method}"] = coords_arr

        # Build scatter plot and set on session.pending_figure
        fig, ax = plt.subplots(figsize=(8, 6))
        df_plot = pd.DataFrame({
            "Dim 1": coords_arr[:, 0],
            "Dim 2": coords_arr[:, 1],
            "Category": labels,
        })

        sns.scatterplot(
            data=df_plot,
            x="Dim 1",
            y="Dim 2",
            hue="Category",
            style="Category",
            s=80,
            ax=ax,
        )
        ax.set_title(f"{method.upper()} Dimensionality Reduction (n_components={n_components})")
        plt.tight_layout()

        session.pending_figure = fig

        # Round coordinates for JSON output
        coords_list = [
            [round(float(val), 4) for val in row] for row in coords_arr
        ]

        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": n_components,
            "coordinates": coords_list,
        }

        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {
                "method": method,
                "n_components": n_components,
                "perplexity": perplexity,
                "n_neighbors": n_neighbors,
                "random_state": random_state,
            },
            summary,
            full_report=df_plot,
        )

        return summary

    @tool
    def binarize_labels_tool(
        column_name: str = "category_name",
    ) -> Dict[str, Any]:
        """One-hot encodes a categorical label column in session.dataframe into a binary matrix.

        Uses sklearn OneHotEncoder to ensure consistent multi-column 0/1 indicator matrices
        (avoiding 1D binary vectors for 2-class problems). Stores the binarized matrix
        and fitted encoder in session.artifacts.

        Args:
            column_name: Column name in session.dataframe to binarize (default: "category_name").

        Returns:
            Dict containing result_id, classes, shape [n_samples, n_classes], or an error dict if validation fails.
        """
        if session.dataframe is None:
            return {
                "status": "error",
                "message": "No dataset in session. Load data first.",
            }

        if column_name not in session.dataframe.columns:
            available_cols = list(session.dataframe.columns)
            return {
                "status": "error",
                "message": f"Column '{column_name}' not found in dataframe. Available columns: {available_cols}",
            }

        series = session.dataframe[column_name]

        if series.isna().any():
            n_missing = int(series.isna().sum())
            return {
                "status": "error",
                "message": f"Column '{column_name}' contains {n_missing} missing (NaN) values. Clean missing values first.",
            }

        labels_arr = series.to_numpy().reshape(-1, 1)

        encoder = OneHotEncoder(sparse_output=False, dtype=int)
        binarized_matrix = encoder.fit_transform(labels_arr)

        raw_categories = encoder.categories_[0]
        classes = [str(cat) for cat in raw_categories]

        df_binarized = pd.DataFrame(
            binarized_matrix,
            columns=classes,
            index=session.dataframe.index,
        )

        # Store matrix and fitted encoder in session.artifacts
        session.artifacts["binarized_labels"] = binarized_matrix
        session.artifacts["binarized_labels_df"] = df_binarized
        session.artifacts["binarized_labels_encoder"] = encoder

        result_id = session.next_result_id("binarize_labels")
        shape = [int(dim) for dim in binarized_matrix.shape]

        summary = {
            "result_id": result_id,
            "classes": classes,
            "shape": shape,
        }

        session.store_result(
            "binarize_labels_tool",
            {"column_name": column_name},
            summary,
            full_report=df_binarized,
        )

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
