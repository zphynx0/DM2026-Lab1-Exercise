import pytest
import numpy as np
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_pca_known_answer_7():
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

    # Get reduce_dimensions_tool
    reduction_tools = make_reduction_tools(session)
    reduce_dims_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Invoke PCA reduction
    res = reduce_dims_tool.invoke({"method": "pca", "n_components": 2})

    # Expected PCA coordinates from Known Answer #7
    expected_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ]

    actual_coords = res["coordinates"]
    assert len(actual_coords) == 8

    for doc_idx, (act_row, exp_row) in enumerate(zip(actual_coords, expected_coords)):
        assert pytest.approx(act_row[0], abs=1e-3) == exp_row[0], f"Doc {doc_idx} PC1 mismatch"
        assert pytest.approx(act_row[1], abs=1e-3) == exp_row[1], f"Doc {doc_idx} PC2 mismatch"

    # Expected explained variance ratio
    expected_var_ratio = [0.7532, 0.1965]
    actual_var_ratio = res["explained_variance_ratio"]

    for i, exp_val in enumerate(expected_var_ratio):
        assert pytest.approx(actual_var_ratio[i], abs=1e-3) == exp_val, f"PC{i+1} variance ratio mismatch"

    # Verify scatter plot was set on session.pending_figure
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)


def test_reduce_dimensions_tsne_known_answer_8():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dims_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Invoke t-SNE reduction with perplexity=2, random_state=42
    res = reduce_dims_tool.invoke({
        "method": "tsne",
        "n_components": 2,
        "perplexity": 2,
        "random_state": 42
    })

    actual_coords = res["coordinates"]
    assert len(actual_coords) == 8

    # Verify that duplicate documents (Doc 5 and Doc 6) map to identical t-SNE coordinates
    assert pytest.approx(actual_coords[5][0], abs=1e-3) == actual_coords[6][0]
    assert pytest.approx(actual_coords[5][1], abs=1e-3) == actual_coords[6][1]

    # Verify return shape (8 rows, 2 columns)
    assert all(len(row) == 2 for row in actual_coords)

    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)


def test_reduce_dimensions_umap_known_answer_9():
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

    reduction_tools = make_reduction_tools(session)
    reduce_dims_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Invoke UMAP reduction with n_neighbors=3, random_state=42
    res = reduce_dims_tool.invoke({
        "method": "umap",
        "n_components": 2,
        "n_neighbors": 3,
        "random_state": 42
    })

    # Expected UMAP coordinates from Known Answer #9
    expected_coords = [
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622],
    ]

    actual_coords = res["coordinates"]
    assert len(actual_coords) == 8

    for doc_idx, (act_row, exp_row) in enumerate(zip(actual_coords, expected_coords)):
        assert pytest.approx(act_row[0], abs=1e-1) == exp_row[0], f"Doc {doc_idx} UMAP Dim 1 mismatch"
        assert pytest.approx(act_row[1], abs=1e-1) == exp_row[1], f"Doc {doc_idx} UMAP Dim 2 mismatch"

    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)
