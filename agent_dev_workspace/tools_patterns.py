import os
import tempfile
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Literal
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from langchain_core.tools import tool

import PAMI.extras.convert.DF2DB as db_converter
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth


def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: Literal["variance", "tfidf", "term_frequency"] = "variance",
        algorithm: Literal["fpgrowth", "topk", "maxfpgrowth"] = "fpgrowth",
        min_sup: Optional[int] = None,
        k: Optional[int] = None
    ) -> Dict[str, Any]:
        """Mine frequent word patterns for a specific category using PAMI.

        Filters category text using English stopwords and vocabulary percentile filtering,
        then runs frequent pattern mining (FPGrowth, Top-K, or MaxFPGrowth).

        Args:
            category_name: Name of the category to filter and mine documents for.
            filtering_method: Vocabulary selection method ('variance', 'tfidf', or 'term_frequency').
            algorithm: PAMI mining algorithm ('fpgrowth', 'topk', or 'maxfpgrowth').
            min_sup: Minimum support count (required for 'fpgrowth' and 'maxfpgrowth').
            k: Top-K pattern limit (required for 'topk').

        Returns:
            Dict containing result_id, category_name, filtering_method, algorithm,
            number of terms kept, and mined patterns list.
        """
        if session.dataframe is None or session.dataframe.empty:
            return {"error": "No dataframe loaded in session."}

        # 1. Filter dataframe to the target category and extract text
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name]
        text_series = df_cat["text"].dropna().astype(str)
        text_series = text_series[text_series.str.strip() != ""]

        if text_series.empty:
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "terms_kept": 0,
                "patterns": [],
                "note": "no non-empty text found for this category"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 2. Fresh per-category CountVectorizer with English stopwords
        vec = CountVectorizer(stop_words="english")
        dtm_cat = vec.fit_transform(text_series)
        feature_names = np.array(vec.get_feature_names_out())

        if feature_names.size == 0:
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "terms_kept": 0,
                "patterns": [],
                "note": "no terms survived stopword removal"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        dtm_dense = dtm_cat.toarray()

        # 3. Apply vocabulary filtering method
        if filtering_method == "variance":
            variances = np.var(dtm_dense, axis=0)
            p5, p95 = np.percentile(variances, 5), np.percentile(variances, 95)
            keep_mask = (variances >= p5) & (variances <= p95)
        elif filtering_method == "term_frequency":
            counts = np.sum(dtm_dense, axis=0)
            p5, p95 = np.percentile(counts, 5), np.percentile(counts, 95)
            keep_mask = (counts >= p5) & (counts <= p95)
        elif filtering_method == "tfidf":
            tfidf_mat = TfidfTransformer().fit_transform(dtm_cat).toarray()
            mean_tfidf = np.mean(tfidf_mat, axis=0)
            p20 = np.percentile(mean_tfidf, 20)
            keep_mask = mean_tfidf >= p20
        else:
            return {"error": f"Unknown filtering_method: {filtering_method}"}

        surviving_terms = feature_names[keep_mask]
        surviving_matrix = dtm_dense[:, keep_mask]

        # 4. Handle case where 0 terms survive
        if len(surviving_terms) == 0:
            res_id = session.next_result_id("patterns")
            summary = {
                "result_id": res_id,
                "category_name": category_name,
                "filtering_method": filtering_method,
                "algorithm": algorithm,
                "terms_kept": 0,
                "patterns": [],
                "note": "no terms survived filtering"
            }
            session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)
            return summary

        # 5. Convert surviving terms to DataFrame and transactional database
        df_surviving = pd.DataFrame(surviving_matrix, columns=surviving_terms)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            db_file = os.path.join(tmpdir, "transactional.db")
            converter = db_converter.DF2DB(df_surviving)
            converter.convert2TransactionalDatabase(db_file, ">=", 1)

            patterns = []
            # 6. Run selected PAMI algorithm
            if algorithm == "fpgrowth":
                if min_sup is None:
                    return {"error": "min_sup parameter is required for fpgrowth algorithm"}
                obj = FPGrowth.FPGrowth(db_file, minSup=min_sup)
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns.append({
                            "pattern": row["Patterns"],
                            "support": int(row["Support"])
                        })

            elif algorithm == "topk":
                if k is None:
                    return {"error": "k parameter is required for topk algorithm"}
                obj = FAE.FAE(db_file, k=k)
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns.append({
                            "pattern": row["Patterns"],
                            "support": int(row["Support"])
                        })

            elif algorithm == "maxfpgrowth":
                if min_sup is None:
                    return {"error": "min_sup parameter is required for maxfpgrowth algorithm"}
                obj = MaxFPGrowth.MaxFPGrowth(db_file, minSup=min_sup)
                obj.startMine()
                df_res = obj.getPatternsAsDataFrame()
                if not df_res.empty:
                    for _, row in df_res.iterrows():
                        patterns.append({
                            "pattern": row["Patterns"],
                            "support": int(row["Support"])
                        })
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

        # 7. Store result and save artifact
        res_id = session.next_result_id("patterns")
        summary = {
            "result_id": res_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "terms_kept": len(surviving_terms),
            "patterns": patterns
        }

        artifact_key = f"patterns_{category_name}_{filtering_method}_{algorithm}"
        session.artifacts[artifact_key] = patterns
        session.store_result("mine_patterns_tool", {"category_name": category_name}, summary)

        return summary

    return [mine_patterns_tool]
