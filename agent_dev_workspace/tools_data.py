import os
from typing import List, Dict, Any, Literal, Optional
import nltk
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def _ensure_nltk_punkt():
    """Lazy helper to check/download NLTK punkt tokenizer data."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        try:
            nltk.download("punkt", quiet=True)
        except Exception as e:
            raise RuntimeError(f"Failed to download NLTK 'punkt' package: {e}")

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception as e:
            raise RuntimeError(f"Failed to download NLTK 'punkt_tab' package: {e}")


def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = ".") -> Dict[str, Any]:
        """Lists files and directories in the specified subdirectory of the repository.

        Args:
            subdirectory (str): Subdirectory path relative to the repository root. Defaults to '.' (root).

        Returns:
            dict: Summary containing result_id, target subdirectory, directories list, and files list.
        """
        # Resolve target directory relative to current working directory
        target_path = os.path.abspath(subdirectory)
        
        if not os.path.exists(target_path):
            result_id = session.next_result_id("list_files")
            summary = {
                "result_id": result_id,
                "subdirectory": subdirectory,
                "error": f"Path '{subdirectory}' does not exist."
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary

        directories = []
        files = []

        try:
            entries = os.listdir(target_path)
            for entry in sorted(entries):
                # Skip hidden files and special directories
                if entry.startswith(".") or entry.startswith("__"):
                    continue
                
                full_path = os.path.join(target_path, entry)
                if os.path.isdir(full_path):
                    directories.append(entry)
                elif os.path.isfile(full_path):
                    files.append(entry)
        except Exception as e:
            result_id = session.next_result_id("list_files")
            summary = {
                "result_id": result_id,
                "subdirectory": subdirectory,
                "error": str(e)
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary

        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def tokenize_tool() -> Dict[str, Any]:
        """Tokenizes each document's text into lowercased unigrams using nltk.word_tokenize.

        Reads session.dataframe["text"], lowercases text, and tokenizes into unigrams.
        Stores token lists in session.dataframe["unigrams"] and session.artifacts["unigrams"].

        Returns:
            Dict containing result_id, sample_tokens (capped to 20 from first non-empty doc), and total_documents.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("tokenize")
            summary = {
                "result_id": result_id,
                "error": "No dataset loaded in session. Load a dataset first."
            }
            session.store_result("tokenize_tool", {}, summary)
            return summary

        if "text" not in session.dataframe.columns:
            result_id = session.next_result_id("tokenize")
            summary = {
                "result_id": result_id,
                "error": "Column 'text' missing from session.dataframe."
            }
            session.store_result("tokenize_tool", {}, summary)
            return summary

        # Lazy check for NLTK tokenizer resources
        _ensure_nltk_punkt()

        unigrams_list = []
        sample_tokens = []
        found_sample = False

        for text in session.dataframe["text"]:
            if text is None or not isinstance(text, str) or text.strip() == "":
                unigrams_list.append([])
            else:
                tokens = nltk.word_tokenize(text.lower())
                unigrams_list.append(tokens)
                if not found_sample and len(tokens) > 0:
                    sample_tokens = tokens[:20]
                    found_sample = True

        # Store in session.dataframe["unigrams"] and session.artifacts["unigrams"]
        session.dataframe["unigrams"] = unigrams_list
        session.artifacts["unigrams"] = unigrams_list

        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "total_documents": len(unigrams_list),
            "sample_tokens": sample_tokens,
        }

        # Store result without full_report
        session.store_result(
            "tokenize_tool",
            {},
            summary,
        )

        return summary

    @tool
    def inspect_data_tool(
        n: int = 5,
        position: Literal["head", "tail", "sample"] = "head",
    ) -> Dict[str, Any]:
        """Peeks at a specified number of rows in session.dataframe from the head, tail, or random sample.

        Args:
            n (int): Number of rows to inspect. Default: 5.
            position (Literal["head", "tail", "sample"]): Strategy for row selection ("head", "tail", or "sample"). Default: "head".

        Returns:
            Dict containing result_id, columns, total_rows, inspected_rows, position, and records list.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("inspect_data")
            summary = {
                "result_id": result_id,
                "error": "No dataset loaded in session. Load a dataset first.",
            }
            session.store_result("inspect_data_tool", {"n": n, "position": position}, summary)
            return summary

        if n <= 0:
            result_id = session.next_result_id("inspect_data")
            summary = {
                "result_id": result_id,
                "error": f"Parameter 'n' must be a positive integer, got {n}.",
            }
            session.store_result("inspect_data_tool", {"n": n, "position": position}, summary)
            return summary

        df = session.dataframe
        total_rows = len(df)

        if position == "head":
            sub_df = df.head(n)
        elif position == "tail":
            sub_df = df.tail(n)
        elif position == "sample":
            sub_df = df.sample(min(n, total_rows), random_state=42)
        else:
            result_id = session.next_result_id("inspect_data")
            summary = {
                "result_id": result_id,
                "error": f"Unsupported position '{position}'. Must be 'head', 'tail', or 'sample'.",
            }
            session.store_result("inspect_data_tool", {"n": n, "position": position}, summary)
            return summary

        # Convert records cleanly to dicts handling timestamps / NaN values safely
        records = sub_df.fillna("").to_dict(orient="records")

        result_id = session.next_result_id("inspect_data")
        summary = {
            "result_id": result_id,
            "columns": list(df.columns),
            "total_rows": total_rows,
            "inspected_rows": len(records),
            "position": position,
            "records": records,
        }

        session.store_result(
            "inspect_data_tool",
            {"n": n, "position": position},
            summary,
            full_report=sub_df,
        )

        return summary

    @tool
    def check_missing_tool() -> Dict[str, Any]:
        """Check the working dataset for missing, null, or empty text entries.

        Inspects session.dataframe['text'] for missing values (NaN/null) and empty
        string entries (including whitespace-only strings).

        Returns:
            Dict containing result_id, total_rows, missing_count, null_count,
            empty_count, and sample missing_indices (capped at 20).
        """
        if session.dataframe is None:
            result_id = session.next_result_id("check_missing")
            summary = {
                "result_id": result_id,
                "error": "No dataset loaded in session. Load a dataset first.",
            }
            session.store_result("check_missing_tool", {}, summary)
            return summary

        df = session.dataframe
        if "text" not in df.columns:
            result_id = session.next_result_id("check_missing")
            summary = {
                "result_id": result_id,
                "error": "Column 'text' missing from session.dataframe.",
            }
            session.store_result("check_missing_tool", {}, summary)
            return summary

        # Vectorized missing/null and empty checks
        null_mask = df["text"].isna()
        empty_mask = ~null_mask & (df["text"].astype(str).str.strip() == "")
        missing_mask = null_mask | empty_mask

        null_count = int(null_mask.sum())
        empty_count = int(empty_mask.sum())
        missing_count = int(missing_mask.sum())
        total_rows = len(df)

        all_missing_indices = [int(i) for i in df.index[missing_mask].tolist()]
        sample_missing_indices = all_missing_indices[:20]

        # Store full missing indices list in artifacts
        session.artifacts["missing_text_indices"] = all_missing_indices

        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "missing_count": missing_count,
            "null_count": null_count,
            "empty_count": empty_count,
            "missing_indices": sample_missing_indices,
        }

        full_report = pd.DataFrame({
            "index": all_missing_indices,
            "is_null": [bool(null_mask.loc[idx]) for idx in all_missing_indices],
            "is_empty": [bool(empty_mask.loc[idx]) for idx in all_missing_indices],
        }) if all_missing_indices else None

        session.store_result(
            "check_missing_tool",
            {},
            summary,
            full_report=full_report,
        )

        return summary

    @tool
    def check_duplicates_tool(
        drop: bool = False,
        subset: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Checks for (and optionally drops) duplicate documents in session.dataframe.

        Args:
            drop (bool): If True, drops all copies of duplicated rows in-place (keep=False). Default: False.
            subset (Optional[List[str]]): List of column names to check for duplicates. Defaults to ["text"].

        Returns:
            Dict containing result_id, total_rows, duplicate_count, extra_copies,
            dropped, remaining_rows, and duplicate_indices.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("check_duplicates")
            summary = {
                "result_id": result_id,
                "error": "No dataset loaded in session. Load a dataset first.",
            }
            session.store_result("check_duplicates_tool", {"drop": drop, "subset": subset}, summary)
            return summary

        df = session.dataframe

        # Default subset to ["text"] if present, else default to all columns
        if subset is None:
            subset = ["text"] if "text" in df.columns else list(df.columns)

        # Validate that all requested columns exist in dataframe
        missing_cols = [col for col in subset if col not in df.columns]
        if missing_cols:
            result_id = session.next_result_id("check_duplicates")
            summary = {
                "result_id": result_id,
                "error": f"Subset columns not found in dataframe: {missing_cols}",
            }
            session.store_result("check_duplicates_tool", {"drop": drop, "subset": subset}, summary)
            return summary

        total_rows = len(df)

        # duplicate_count = all duplicated rows including all copies
        all_duplicates_mask = df.duplicated(subset=subset, keep=False)
        duplicate_count = int(all_duplicates_mask.sum())

        # extra_copies = standard pandas default duplicates count (keep="first")
        extra_copies_mask = df.duplicated(subset=subset, keep="first")
        extra_copies = int(extra_copies_mask.sum())

        duplicate_indices = [int(i) for i in df.index[all_duplicates_mask].tolist()]

        full_report = df.loc[all_duplicates_mask].copy() if duplicate_count > 0 else None

        if drop and duplicate_count > 0:
            # Drop ALL copies of duplicated rows matching keep=False
            session.dataframe.drop_duplicates(subset=subset, keep=False, inplace=True)
            dropped = True

            # Clear/update stale artifacts if dataframe rows were dropped
            if "unigrams" in session.artifacts:
                if "unigrams" in session.dataframe.columns:
                    session.artifacts["unigrams"] = session.dataframe["unigrams"].tolist()
                else:
                    session.artifacts.pop("unigrams", None)

            if "missing_text_indices" in session.artifacts and "text" in session.dataframe.columns:
                null_mask = session.dataframe["text"].isna()
                empty_mask = ~null_mask & (session.dataframe["text"].astype(str).str.strip() == "")
                missing_mask = null_mask | empty_mask
                session.artifacts["missing_text_indices"] = [
                    int(i) for i in session.dataframe.index[missing_mask].tolist()
                ]
        else:
            dropped = False

        remaining_rows = len(session.dataframe)

        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "duplicate_count": duplicate_count,
            "extra_copies": extra_copies,
            "dropped": dropped,
            "remaining_rows": remaining_rows,
            "duplicate_indices": duplicate_indices[:20],
        }

        session.store_result(
            "check_duplicates_tool",
            {"drop": drop, "subset": subset},
            summary,
            full_report=full_report,
        )

        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = 42,
        replace: bool = False,
        inplace: bool = False,
    ) -> Dict[str, Any]:
        """Subsamples rows from session.dataframe using pandas sample.

        Args:
            n (Optional[int]): Number of rows to sample (must be > 0).
            frac (Optional[float]): Fraction of rows to sample (must be in 0 < frac <= 1).
            random_state (Optional[int]): Seed for random number generator. Defaults to 42.
            replace (bool): Sample with or without replacement. Defaults to False.
            inplace (bool): If True, replaces session.dataframe with sampled subset. Defaults to False.

        Returns:
            Dict containing result_id, total_rows, sample_size, sampled_indices,
            inplace, and sample records preview.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("sample_data")
            summary = {
                "result_id": result_id,
                "error": "No dataset loaded in session. Load a dataset first.",
            }
            session.store_result(
                "sample_data_tool",
                {"n": n, "frac": frac, "random_state": random_state, "replace": replace, "inplace": inplace},
                summary,
            )
            return summary

        # 1. Validation of inputs up front
        if n is not None and frac is not None:
            result_id = session.next_result_id("sample_data")
            summary = {
                "result_id": result_id,
                "error": "Specify either 'n' or 'frac', not both.",
            }
            session.store_result(
                "sample_data_tool",
                {"n": n, "frac": frac, "random_state": random_state, "replace": replace, "inplace": inplace},
                summary,
            )
            return summary

        if n is None and frac is None:
            result_id = session.next_result_id("sample_data")
            summary = {
                "result_id": result_id,
                "error": "Must specify either 'n' or 'frac'.",
            }
            session.store_result(
                "sample_data_tool",
                {"n": n, "frac": frac, "random_state": random_state, "replace": replace, "inplace": inplace},
                summary,
            )
            return summary

        if n is not None and n <= 0:
            result_id = session.next_result_id("sample_data")
            summary = {
                "result_id": result_id,
                "error": f"'n' must be a positive integer, got {n}.",
            }
            session.store_result(
                "sample_data_tool",
                {"n": n, "frac": frac, "random_state": random_state, "replace": replace, "inplace": inplace},
                summary,
            )
            return summary

        if frac is not None and (frac <= 0 or frac > 1.0):
            result_id = session.next_result_id("sample_data")
            summary = {
                "result_id": result_id,
                "error": f"'frac' must be in range 0 < frac <= 1.0, got {frac}.",
            }
            session.store_result(
                "sample_data_tool",
                {"n": n, "frac": frac, "random_state": random_state, "replace": replace, "inplace": inplace},
                summary,
            )
            return summary

        if replace and inplace:
            result_id = session.next_result_id("sample_data")
            summary = {
                "result_id": result_id,
                "error": "Cannot use replace=True with inplace=True (creates duplicate index labels).",
            }
            session.store_result(
                "sample_data_tool",
                {"n": n, "frac": frac, "random_state": random_state, "replace": replace, "inplace": inplace},
                summary,
            )
            return summary

        df = session.dataframe
        total_rows = len(df)

        try:
            sample_df = df.sample(
                n=n,
                frac=frac,
                random_state=random_state,
                replace=replace,
            )
        except Exception as e:
            result_id = session.next_result_id("sample_data")
            summary = {
                "result_id": result_id,
                "error": f"Sampling failed: {str(e)}",
            }
            session.store_result(
                "sample_data_tool",
                {"n": n, "frac": frac, "random_state": random_state, "replace": replace, "inplace": inplace},
                summary,
            )
            return summary

        all_sampled_indices = [int(i) for i in sample_df.index.tolist()]
        sample_size = len(sample_df)

        if inplace:
            session.dataframe = sample_df.copy()
            # Clear / update all stale row-tied artifacts
            if "unigrams" in session.artifacts:
                if "unigrams" in session.dataframe.columns:
                    session.artifacts["unigrams"] = session.dataframe["unigrams"].tolist()
                else:
                    session.artifacts.pop("unigrams", None)

            if "missing_text_indices" in session.artifacts and "text" in session.dataframe.columns:
                null_mask = session.dataframe["text"].isna()
                empty_mask = ~null_mask & (session.dataframe["text"].astype(str).str.strip() == "")
                missing_mask = null_mask | empty_mask
                session.artifacts["missing_text_indices"] = [
                    int(i) for i in session.dataframe.index[missing_mask].tolist()
                ]

            # Clear any stale binarized labels or row-aligned feature matrix artifacts
            session.artifacts.pop("binarized_labels", None)
            session.feature_matrix = None
            session.feature_names = None
            session.labels = None

        # Build preview records from head(20) of sampled subset
        records_preview = sample_df.head(20).fillna("").to_dict(orient="records")

        result_id = session.next_result_id("sample_data")
        summary = {
            "result_id": result_id,
            "total_rows": total_rows,
            "sample_size": sample_size,
            "sampled_indices": all_sampled_indices[:20],
            "inplace": inplace,
            "records": records_preview,
        }

        session.store_result(
            "sample_data_tool",
            {"n": n, "frac": frac, "random_state": random_state, "replace": replace, "inplace": inplace},
            summary,
            full_report=sample_df,
        )

        return summary

    @tool
    def describe_data_tool() -> Dict[str, Any]:
        """Calculates descriptive statistics on document text length and generates a boxplot by category.

        Computes character length for document texts, calculates overall and per-category
        summary statistics (.describe()), generates a boxplot set on session.pending_figure,
        and returns JSON-serializable stats.

        Returns:
            Dict containing result_id, overall_stats, category_stats, and text_lengths list.
        """
        if session.dataframe is None:
            result_id = session.next_result_id("describe_data")
            summary = {
                "result_id": result_id,
                "error": "No dataset loaded in session. Load a dataset first.",
            }
            session.store_result("describe_data_tool", {}, summary)
            return summary

        df = session.dataframe
        required_cols = ["text", "category_name"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            result_id = session.next_result_id("describe_data")
            summary = {
                "result_id": result_id,
                "error": f"Missing required columns in session.dataframe: {missing_cols}",
            }
            session.store_result("describe_data_tool", {}, summary)
            return summary

        # Local series for text length, fillna with empty string so NaN text length is 0
        text_length_series = df["text"].fillna("").astype(str).str.len()
        text_lengths = [int(l) for l in text_length_series.tolist()]

        # Helper to convert pandas .describe() Series to JSON-safe dict
        def _series_to_stats(s: pd.Series) -> Dict[str, Any]:
            desc = s.describe()
            std_val = desc["std"]
            return {
                "count": int(desc["count"]),
                "mean": round(float(desc["mean"]), 4),
                "std": round(float(std_val), 4) if pd.notna(std_val) else None,
                "min": float(desc["min"]),
                "25%": float(desc["25%"]),
                "50%": float(desc["50%"]),
                "75%": float(desc["75%"]),
                "max": float(desc["max"]),
            }

        overall_stats = _series_to_stats(text_length_series)

        # Per-category summary statistics
        temp_df = pd.DataFrame({
            "text_length": text_length_series,
            "category_name": df["category_name"],
        })

        category_stats = {}
        for cat_name, group in temp_df.groupby("category_name"):
            category_stats[str(cat_name)] = _series_to_stats(group["text_length"])

        # Build Seaborn boxplot grouped by category_name
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(
            data=temp_df,
            x="category_name",
            y="text_length",
            ax=ax,
            palette="Set2",
        )
        ax.set_title("Document Text Length by Category")
        ax.set_xlabel("Category")
        ax.set_ylabel("Text Length (Characters)")

        # Assign figure to session.pending_figure and close figure
        session.pending_figure = fig
        plt.close(fig)

        # Save text lengths in artifacts without polluting session.dataframe
        session.artifacts["text_lengths"] = text_lengths

        result_id = session.next_result_id("describe_data")
        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "category_stats": category_stats,
            "text_lengths": text_lengths,
        }

        session.store_result(
            "describe_data_tool",
            {},
            summary,
            full_report=temp_df,
        )

        return summary

    return [
        list_files_tool,
        tokenize_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
