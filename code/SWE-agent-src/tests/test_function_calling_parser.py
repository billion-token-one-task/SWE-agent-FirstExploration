from sweagent.tools.commands import Argument, Command
from sweagent.tools.parsing import FunctionCallingParser


BASH_COMMAND = Command(
    name="bash",
    docstring="Run a bash command.",
    arguments=[
        Argument(name="command", type="string", description="Command to run.", required=True),
    ],
)

STR_REPLACE_EDITOR_COMMAND = Command(
    name="str_replace_editor",
    docstring="View or edit a file.",
    arguments=[
        Argument(name="command", type="string", description="Editor action.", required=True),
        Argument(name="path", type="string", description="Target path.", required=True),
        Argument(name="file_text", type="string", description="New file contents.", required=False),
        Argument(name="old_str", type="string", description="Text to replace.", required=False),
        Argument(name="new_str", type="string", description="Replacement text.", required=False),
        Argument(name="insert_line", type="integer", description="Insert before line number.", required=False),
        Argument(name="view_range", type="array", description="Optional line range.", required=False),
    ],
)


def test_function_calling_parser_prefers_candidate_matching_bash_schema():
    parser = FunctionCallingParser()
    tool_call = {
        "function": {
            "name": "bash",
            "arguments": '{"command":"view","path":"/testbed"}{"command":"cd /testbed && grep -R foo ."}',
        }
    }

    action = parser._parse_tool_call(tool_call, [BASH_COMMAND])

    assert action == "bash cd /testbed && grep -R foo ."



def test_function_calling_parser_recovers_editor_view_call_from_concatenated_json():
    parser = FunctionCallingParser()
    tool_call = {
        "function": {
            "name": "str_replace_editor",
            "arguments": '{"command":"cd /testbed && find . -name settings.py"}{"command":"view","path":"/testbed"}',
        }
    }

    action = parser._parse_tool_call(tool_call, [STR_REPLACE_EDITOR_COMMAND])

    assert action == "str_replace_editor view /testbed"
