import pytest
import matplotlib.figure
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


# def test_term_frequency_tool_known_answer_14():
#     # Single session setup
#     session = SessionState()

#     # Load premade load_dataset_tool
#     premade_tools, _ = load_workspace_tools("premade_tools", session)
#     load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

#     # Load tools_dtm tools
#     dtm_tools = make_tools(session)
#     build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
#     term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

#     # Load sample_fixture.csv
#     load_dataset_tool.invoke({
#         "file_path": "agent_dev/sample_fixture.csv",
#         "text_column": "text",
#         "label_column": "label"
#     })

#     # Build DTM first
#     build_dtm_res = build_dtm_tool.invoke({})
#     assert build_dtm_res["n_terms"] == 5

#     # Run term_frequency_tool
#     tf_res = term_frequency_tool.invoke({})

#     # Check against Known Answer 14 in TEST_FIXTURE.md
#     expected_frequencies = {
#         "alpha": 6,
#         "always": 7,
#         "beta": 3,
#         "delta": 4,
#         "gamma": 7,
#     }

#     assert "term_frequencies" in tf_res
#     assert tf_res["term_frequencies"] == expected_frequencies

def test_term_frequency_tool_known_answer_14():
    # Single session setup
    session = SessionState()
 
    # Load premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
 
    # Load tools_dtm tools
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
 
    # Load sample_fixture.csv
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
 
    # Build DTM first
    build_dtm_res = build_dtm_tool.invoke({})
    assert build_dtm_res["n_terms"] == 5
 
    # Run term_frequency_tool
    tf_res = term_frequency_tool.invoke({})
 
    # Check against Known Answer 14 in TEST_FIXTURE.md
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }
 
    # term_frequency_tool's documented shape: result_id, total_terms, top_terms
    assert tf_res["total_terms"] == 5
    assert tf_res["top_terms"] == expected_frequencies
 
 
def test_term_frequency_tool_top_n_limits_output():
    session = SessionState()
 
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
 
    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")
 
    build_dtm_tool.invoke({})
 
    # Known Answer 14: "always" and "gamma" tie at 7, everything else is lower
    res = term_frequency_tool.invoke({"top_n": 2})
    assert res["total_terms"] == 5
    assert res["top_terms"] == {"always": 7, "gamma": 7}


def test_dtm_heatmap_tool_known_answer_21():
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({})

    # Test full matrix slice (default n_terms=20, n_documents=20 exceeds 5 terms and 8 docs)
    res_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    expected_terms = ["alpha", "always", "beta", "delta", "gamma"]
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0],
    ]

    assert res_full["terms"] == expected_terms
    assert res_full["matrix"] == expected_full_matrix
    assert session.pending_figure is not None
    assert isinstance(session.pending_figure, matplotlib.figure.Figure)

    # Test custom positional slice (n_terms=2, n_documents=3)
    res_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})

    expected_sliced_terms = ["alpha", "always"]
    expected_sliced_matrix = [
        [3, 1],
        [2, 1],
        [1, 1],
    ]

    assert res_slice["terms"] == expected_sliced_terms
    assert res_slice["matrix"] == expected_sliced_matrix
