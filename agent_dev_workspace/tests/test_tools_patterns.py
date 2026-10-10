import pytest
import pandas as pd
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools


@pytest.fixture
def session_with_data():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    
    # Load test fixture dataset
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session


def test_mine_patterns_fpgrowth_tfidf(session_with_data):
    """Test known answer #17: tfidf filtering + fpgrowth on catA and catB."""
    tools = make_tools(session_with_data)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # catA with min_sup=3
    res_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 3
    })
    assert res_catA["terms_kept"] == 1
    assert len(res_catA["patterns"]) == 1
    assert res_catA["patterns"][0]["pattern"] == "alpha"
    assert res_catA["patterns"][0]["support"] == 3

    # catB with min_sup=4
    res_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 4
    })
    assert res_catB["terms_kept"] == 1
    assert len(res_catB["patterns"]) == 1
    assert res_catB["patterns"][0]["pattern"] == "gamma"
    assert res_catB["patterns"][0]["support"] == 4


def test_mine_patterns_topk_and_maxfpgrowth(session_with_data):
    """Test known answer #20: topk and maxfpgrowth algorithms."""
    tools = make_tools(session_with_data)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    # topk algorithm with k=1 on catA
    res_topk = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk["terms_kept"] == 1
    assert len(res_topk["patterns"]) == 1
    assert res_topk["patterns"][0]["pattern"] == "alpha"
    assert res_topk["patterns"][0]["support"] == 3

    # maxfpgrowth algorithm with min_sup=1 on catA (returns 0 patterns per PAMI single-term behavior)
    res_max = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max["terms_kept"] == 1
    assert len(res_max["patterns"]) == 0


def test_mine_patterns_zero_surviving_terms(session_with_data):
    """Test edge case where 0 terms survive variance filtering."""
    tools = make_tools(session_with_data)
    mine_patterns_tool = next(t for t in tools if t.name == "mine_patterns_tool")

    res = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res["terms_kept"] == 0
    assert res["patterns"] == []
    assert res["note"] == "no terms survived filtering"
