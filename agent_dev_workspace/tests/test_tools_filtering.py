import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_correlation_filtering_known_answer_6():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # Build DTM using student's build_dtm_tool
    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    # Get filtering tools
    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Run pearson_filter_tool for target_class="catB"
    pearson_res = pearson_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    pearson_corrs = pearson_res["correlations"]

    # Known answer 6 expected Pearson correlation values
    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, expected_val in expected_pearson.items():
        assert term in pearson_corrs, f"Term '{term}' missing in pearson correlation output"
        assert pytest.approx(pearson_corrs[term], abs=1e-3) == expected_val, (
            f"Pearson r for '{term}' mismatch: expected {expected_val}, got {pearson_corrs[term]}"
        )

    # Run spearman_filter_tool for target_class="catB"
    spearman_res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})
    spearman_corrs = spearman_res["correlations"]

    # Known answer 6 expected Spearman correlation values
    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for term, expected_val in expected_spearman.items():
        assert term in spearman_corrs, f"Term '{term}' missing in spearman correlation output"
        assert pytest.approx(spearman_corrs[term], abs=1e-3) == expected_val, (
            f"Spearman r for '{term}' mismatch: expected {expected_val}, got {spearman_corrs[term]}"
        )
