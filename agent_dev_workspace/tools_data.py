import os
from typing import List, Dict, Any
from langchain_core.tools import tool

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

    return [list_files_tool]
