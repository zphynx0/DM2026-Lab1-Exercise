import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_list_files_tool_newdataset():
    # Known Answer #11: list_files_tool(subdirectory="newdataset")
    # expected directories: [], files: ["Reddit-stock-sentiment.csv"]
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    res = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert "result_id" in res
    assert res["subdirectory"] == "newdataset"
    assert res["directories"] == []
    assert res["files"] == ["Reddit-stock-sentiment.csv"]


def test_tokenize_tool_known_answer_10():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")

    res = tokenize_tool.invoke({})

    assert res["total_documents"] == 8
    assert res["sample_tokens"] == ["always", "alpha", "alpha", "alpha", "beta"]

    # Verify unigrams in session.dataframe and session.artifacts
    assert "unigrams" in session.dataframe.columns
    assert "unigrams" in session.artifacts

    expected_unigrams = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        [],  # Row 7 is empty text
    ]

    actual_unigrams = session.dataframe["unigrams"].tolist()
    assert actual_unigrams == expected_unigrams
    assert session.artifacts["unigrams"] == expected_unigrams


def test_inspect_data_tool():
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    inspect_data_tool = next(t for t in data_tools if t.name == "inspect_data_tool")

    # 1. Test error when n is non-positive
    err_res = inspect_data_tool.invoke({"n": 0})
    assert "error" in err_res
    assert "must be a positive integer" in err_res["error"]

    # 2. Test default inspection (head, n=3)
    head_res = inspect_data_tool.invoke({"n": 3, "position": "head"})
    assert head_res["total_rows"] == 8
    assert head_res["inspected_rows"] == 3
    assert head_res["position"] == "head"
    assert len(head_res["records"]) == 3
    assert head_res["records"][0]["text"] == "always alpha alpha alpha beta"

    # 3. Test tail inspection (tail, n=2)
    tail_res = inspect_data_tool.invoke({"n": 2, "position": "tail"})
    assert tail_res["inspected_rows"] == 2
    assert tail_res["position"] == "tail"
    # Row 7 is empty text
    assert tail_res["records"][1]["text"] == ""

    # 4. Test sample inspection (sample, n=4)
    sample_res = inspect_data_tool.invoke({"n": 4, "position": "sample"})
    assert sample_res["inspected_rows"] == 4
    assert sample_res["position"] == "sample"


def test_check_missing_tool_known_answer_2():
    """Test check_missing_tool against TEST_FIXTURE.md known answer #2."""
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    check_missing_tool = next(t for t in data_tools if t.name == "check_missing_tool")

    res = check_missing_tool.invoke({})

    # Known answer #2: exactly 1 row has missing/empty text in the 8-row fixture (row index 7)
    assert res["total_rows"] == 8
    assert res["missing_count"] == 1
    assert res["null_count"] == 0
    assert res["empty_count"] == 1
    assert res["missing_indices"] == [7]
    assert session.artifacts.get("missing_text_indices") == [7]


def test_check_duplicates_tool_known_answer_3():
    """Test check_duplicates_tool against TEST_FIXTURE.md known answer #3."""
    # 1. Test drop=False (inspection without modifying dataframe)
    session_inspect = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session_inspect)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools_inspect = make_tools(session_inspect)
    check_dup_tool_inspect = next(t for t in data_tools_inspect if t.name == "check_duplicates_tool")

    res_inspect = check_dup_tool_inspect.invoke({"drop": False})

    assert res_inspect["total_rows"] == 8
    assert res_inspect["duplicate_count"] == 2  # rows 5 and 6 are copies of "always gamma delta"
    assert res_inspect["extra_copies"] == 1
    assert res_inspect["dropped"] is False
    assert res_inspect["remaining_rows"] == 8
    assert res_inspect["duplicate_indices"] == [5, 6]
    assert len(session_inspect.dataframe) == 8

    # 2. Test drop=True (dropping with keep=False leaves exactly 6 rows, removing BOTH rows 5 and 6)
    session_drop = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session_drop)
    load_dataset_tool_drop = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool_drop.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools_drop = make_tools(session_drop)
    check_dup_tool_drop = next(t for t in data_tools_drop if t.name == "check_duplicates_tool")

    res_drop = check_dup_tool_drop.invoke({"drop": True})

    assert res_drop["total_rows"] == 8
    assert res_drop["duplicate_count"] == 2
    assert res_drop["extra_copies"] == 1
    assert res_drop["dropped"] is True
    assert res_drop["remaining_rows"] == 6

    # Verify both row 5 and row 6 are gone from dataframe indices
    remaining_indices = session_drop.dataframe.index.tolist()
    assert 5 not in remaining_indices
    assert 6 not in remaining_indices
    assert remaining_indices == [0, 1, 2, 3, 4, 7]


def test_sample_data_tool_known_answer_13():
    """Test sample_data_tool against TEST_FIXTURE.md known answer #13."""
    # 1. Known Answer #13: n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    session_sample = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session_sample)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session_sample)
    sample_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    res = sample_tool.invoke({"n": 4, "random_state": 42, "inplace": False})

    assert res["total_rows"] == 8
    assert res["sample_size"] == 4
    assert res["sampled_indices"] == [1, 5, 0, 7]
    assert res["inplace"] is False
    # DataFrame in session should be unchanged when inplace=False
    assert len(session_sample.dataframe) == 8

    # 2. Test specifying both n and frac raises an error
    err_both = sample_tool.invoke({"n": 3, "frac": 0.5})
    assert "error" in err_both
    assert "Specify either 'n' or 'frac', not both" in err_both["error"]

    # 3. Test inplace=True updates session.dataframe in place
    res_inplace = sample_tool.invoke({"n": 3, "random_state": 42, "inplace": True})
    assert res_inplace["sample_size"] == 3
    assert res_inplace["inplace"] is True
    assert len(session_sample.dataframe) == 3


def test_describe_data_tool_known_answer_18():
    """Test describe_data_tool against TEST_FIXTURE.md known answer #18."""
    session = SessionState()

    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    describe_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    res = describe_tool.invoke({})

    # Known Answer #18 text lengths: [29, 23, 17, 30, 24, 18, 18, 0]
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    assert res["text_lengths"] == expected_lengths

    # Overall stats assertions (with pytest.approx tolerances)
    overall = res["overall_stats"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0.0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30.0

    # Per-category stats assertions
    cat_stats = res["category_stats"]
    assert "catA" in cat_stats
    assert "catB" in cat_stats

    catA = cat_stats["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-2)  # 12.5 or 12.506 depending on ddof
    assert catA["min"] == 0.0
    assert catA["max"] == 29.0

    catB = cat_stats["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18.0
    assert catB["max"] == 30.0

    # Check matplotlib figure requirement on session.pending_figure
    assert session.pending_figure is not None
