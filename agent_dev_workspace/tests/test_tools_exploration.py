import pytest
import numpy as np
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools


def test_feature_correlation_matrix_known_answer_19():
    session = SessionState()

    # Load premade load_dataset_tool
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

    # Get feature_correlation_matrix_tool
    exploration_tools = make_exploration_tools(session)
    feature_corr_tool = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    # Invoke tool
    res = feature_corr_tool.invoke({})

    # Expected top-variance terms order
    expected_terms = ["alpha", "gamma", "delta", "beta", "always"]
    assert res["terms"] == expected_terms

    # Expected correlation matrix (Known Answer #19)
    expected_corr = [
        [1.0000, -0.5718, -0.6882, 0.8885, 0.2601],
        [-0.5718, 1.0000, 0.8307, -0.6435, 0.3140],
        [-0.6882, 0.8307, 1.0000, -0.7746, 0.3780],
        [0.8885, -0.6435, -0.7746, 1.0000, 0.2928],
        [0.2601, 0.3140, 0.3780, 0.2928, 1.0000],
    ]

    actual_corr = res["correlation_matrix"]
    assert len(actual_corr) == 5

    for i in range(5):
        for j in range(5):
            assert pytest.approx(actual_corr[i][j], abs=1e-3) == expected_corr[i][j], (
                f"Mismatch at ({expected_terms[i]}, {expected_terms[j]}): "
                f"expected {expected_corr[i][j]}, got {actual_corr[i][j]}"
            )

    # Verify plot was set on session.pending_figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)


def test_cosine_similarity_known_answer_16():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({})

    exploration_tools = make_exploration_tools(session)
    cosine_sim_tool = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Test document pairs from Known Answer #16
    pairs_expected = [
        (0, 1, 0.9847),  # Both catA, near-identical
        (0, 3, 0.0909),  # catA vs catB, sharing only "always"
        (3, 4, 0.9847),  # Both catB, near-identical
        (0, 7, 0.0),     # catA vs empty document
    ]

    for idx1, idx2, expected_sim in pairs_expected:
        res = cosine_sim_tool.invoke({"doc_index1": idx1, "doc_index2": idx2})
        assert "cosine_similarity" in res
        assert pytest.approx(res["cosine_similarity"], abs=1e-3) == expected_sim, (
            f"Cosine similarity mismatch for pair ({idx1}, {idx2}): "
            f"expected {expected_sim}, got {res['cosine_similarity']}"
        )
