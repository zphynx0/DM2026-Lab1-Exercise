import pytest
from agent_pipeline.session_state import SessionState
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
