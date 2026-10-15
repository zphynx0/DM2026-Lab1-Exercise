import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_known_answer_5():
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
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    # Run variance_filter_tool with threshold=0.15
    res = variance_filter_tool.invoke({"threshold": 0.15})

    # Assert on res itself
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1
    assert "top_20_kept_by_variance" in res
    assert "top_20_removed_by_variance" in res

    # Read full table from session.results_store[res["result_id"]]["full_report"]
    stored_entry = session.results_store[res["result_id"]]
    full_report = stored_entry["full_report"]

    var_dict = dict(zip(full_report["term"], full_report["variance"]))

    # Known answer 5 expected variances across all 5 terms
    expected_variances = {
        "alpha": 1.1875,
        "always": 0.109375,
        "beta": 0.234375,
        "delta": 0.25,
        "gamma": 1.109375,
    }

    for term, expected_val in expected_variances.items():
        assert term in var_dict, f"Term '{term}' missing in full_report"
        assert pytest.approx(var_dict[term], abs=1e-3) == expected_val, (
            f"Variance for '{term}' mismatch: expected {expected_val}, got {var_dict[term]}"
        )


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

    # Assert on pearson_res itself
    assert pearson_res["n_kept"] == 5
    assert pearson_res["n_removed"] == 0
    assert "top_20_kept_by_abs_r" in pearson_res
    assert "top_20_removed_by_abs_r" in pearson_res

    # Read full table from session.results_store[pearson_res["result_id"]]["full_report"]
    pearson_stored = session.results_store[pearson_res["result_id"]]
    pearson_report = pearson_stored["full_report"]
    pearson_corrs = dict(zip(pearson_report["term"], pearson_report["pearson_r"]))

    expected_pearson = {
        "alpha": -0.6882,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.8307,
    }

    for term, expected_val in expected_pearson.items():
        assert term in pearson_corrs, f"Term '{term}' missing in pearson full_report"
        assert pytest.approx(pearson_corrs[term], abs=1e-3) == expected_val, (
            f"Pearson r for '{term}' mismatch: expected {expected_val}, got {pearson_corrs[term]}"
        )

    # Run spearman_filter_tool for target_class="catB"
    spearman_res = spearman_filter_tool.invoke({"target_class": "catB", "threshold": 0.0})

    # Assert on spearman_res itself
    assert spearman_res["n_kept"] == 5
    assert spearman_res["n_removed"] == 0
    assert "top_20_kept_by_abs_r" in spearman_res
    assert "top_20_removed_by_abs_r" in spearman_res

    # Read full table from session.results_store[spearman_res["result_id"]]["full_report"]
    spearman_stored = session.results_store[spearman_res["result_id"]]
    spearman_report = spearman_stored["full_report"]
    spearman_corrs = dict(zip(spearman_report["term"], spearman_report["spearman_r"]))

    expected_spearman = {
        "alpha": -0.7500,
        "always": 0.3780,
        "beta": -0.7746,
        "delta": 1.0000,
        "gamma": 0.9363,
    }

    for term, expected_val in expected_spearman.items():
        assert term in spearman_corrs, f"Term '{term}' missing in spearman full_report"
        assert pytest.approx(spearman_corrs[term], abs=1e-3) == expected_val, (
            f"Spearman r for '{term}' mismatch: expected {expected_val}, got {spearman_corrs[term]}"
        )
