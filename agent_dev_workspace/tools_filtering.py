from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from scipy import stats
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.15
    ) -> Dict[str, Any]:
        """Flags near-constant terms by variance across documents for removal.

        Computes feature variances across all documents in session.feature_matrix and
        identifies terms meeting or falling below the specified variance threshold.

        Args:
            threshold: Minimum variance threshold required to keep a term (default: 0.15).

        Returns:
            Dict containing result_id, threshold, n_kept, n_removed, top_20_kept_by_variance, and top_20_removed_by_variance.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"status": "error", "message": "No document-term matrix in session. Run build_dtm_tool first."}

        # Convert sparse matrix to array for variance calculation across documents (axis=0)
        dense_matrix = session.feature_matrix.toarray()
        variances_arr = np.var(dense_matrix, axis=0)

        terms = session.feature_names
        var_dict = {term: float(v) for term, v in zip(terms, variances_arr)}

        kept_terms = [term for term, v in var_dict.items() if v >= threshold]
        removed_terms = [term for term, v in var_dict.items() if v < threshold]

        # Prepare full_report DataFrame sorted descending by variance
        # 1st column: term, 2nd column: variance (for visualize_result_tool compatibility)
        sorted_items = sorted(var_dict.items(), key=lambda item: item[1], reverse=True)
        df_report = pd.DataFrame(sorted_items, columns=["term", "variance"])

        result_id = session.next_result_id("variance_filter")
        
        # Keep stored summary and returned dict small
        top_20_kept = [(t, round(v, 6)) for t, v in sorted_items if v >= threshold][:20]
        sorted_removed = sorted([(t, var_dict[t]) for t in removed_terms], key=lambda item: item[1])
        top_20_removed = [(t, round(v, 6)) for t, v in sorted_removed][:20]

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "top_20_kept_by_variance": top_20_kept,
            "top_20_removed_by_variance": top_20_removed,
        }

        # Full per-term table stored in full_report
        session.store_result("variance_filter_tool", {"threshold": threshold}, summary, full_report=df_report)

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """Filters terms based on Pearson correlation with a binary target category label.

        Computes linear correlation r between each term's document frequency count
        and an indicator vector for the target_class (1 if doc in target_class, 0 otherwise).

        Args:
            target_class: Target category label string to correlate features against.
            threshold: Minimum absolute correlation required to keep a term (default: 0.2).

        Returns:
            Dict containing result_id, target_class, threshold, n_kept, n_removed, top_20_kept_by_abs_r, and top_20_removed_by_abs_r.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"status": "error", "message": "No document-term matrix or labels in session. Run build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        binary_target = (session.labels == target_class).astype(int)

        terms = session.feature_names
        corr_dict = {}

        for i, term in enumerate(terms):
            counts = dense_matrix[:, i]
            # Avoid divide-by-zero warnings for zero-variance counts or targets
            if np.std(counts) == 0 or np.std(binary_target) == 0:
                r_val = 0.0
            else:
                r_val, _ = stats.pearsonr(counts, binary_target)
                if np.isnan(r_val):
                    r_val = 0.0
            corr_dict[term] = float(r_val)

        kept_terms = [term for term, r in corr_dict.items() if abs(r) >= threshold]
        removed_terms = [term for term, r in corr_dict.items() if abs(r) < threshold]

        sorted_items = sorted(corr_dict.items(), key=lambda item: item[1], reverse=True)
        df_report = pd.DataFrame(sorted_items, columns=["term", "pearson_r"])

        result_id = session.next_result_id("pearson_filter")

        top_20_kept = sorted(
            ((t, round(r, 6)) for t, r in corr_dict.items() if abs(r) >= threshold),
            key=lambda x: abs(x[1]), reverse=True,
        )[:20]
        top_20_removed = sorted(
            ((t, round(r, 6)) for t, r in corr_dict.items() if abs(r) < threshold),
            key=lambda x: abs(x[1]),
        )[:20]

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "top_20_kept_by_abs_r": top_20_kept,
            "top_20_removed_by_abs_r": top_20_removed,
        }

        # Full per-term table stored in full_report
        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.2
    ) -> Dict[str, Any]:
        """Filters terms based on Spearman rank correlation with a binary target category label.

        Computes monotonic rank correlation r_s between each term's document frequency count
        and an indicator vector for the target_class (1 if doc in target_class, 0 otherwise).

        Args:
            target_class: Target category label string to correlate features against.
            threshold: Minimum absolute correlation required to keep a term (default: 0.2).

        Returns:
            Dict containing result_id, target_class, threshold, n_kept, n_removed, top_20_kept_by_abs_r, and top_20_removed_by_abs_r.
        """
        if session.feature_matrix is None or session.feature_names is None or session.labels is None:
            return {"status": "error", "message": "No document-term matrix or labels in session. Run build_dtm_tool first."}

        dense_matrix = session.feature_matrix.toarray()
        binary_target = (session.labels == target_class).astype(int)

        terms = session.feature_names
        corr_dict = {}

        for i, term in enumerate(terms):
            counts = dense_matrix[:, i]
            if np.std(counts) == 0 or np.std(binary_target) == 0:
                r_val = 0.0
            else:
                r_val, _ = stats.spearmanr(counts, binary_target)
                if np.isnan(r_val):
                    r_val = 0.0
            corr_dict[term] = float(r_val)

        kept_terms = [term for term, r in corr_dict.items() if abs(r) >= threshold]
        removed_terms = [term for term, r in corr_dict.items() if abs(r) < threshold]

        sorted_items = sorted(corr_dict.items(), key=lambda item: item[1], reverse=True)
        df_report = pd.DataFrame(sorted_items, columns=["term", "spearman_r"])

        result_id = session.next_result_id("spearman_filter")

        top_20_kept = sorted(
            ((t, round(r, 6)) for t, r in corr_dict.items() if abs(r) >= threshold),
            key=lambda x: abs(x[1]), reverse=True,
        )[:20]
        top_20_removed = sorted(
            ((t, round(r, 6)) for t, r in corr_dict.items() if abs(r) < threshold),
            key=lambda x: abs(x[1]),
        )[:20]

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "top_20_kept_by_abs_r": top_20_kept,
            "top_20_removed_by_abs_r": top_20_removed,
        }

        # Full per-term table stored in full_report
        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=df_report,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
