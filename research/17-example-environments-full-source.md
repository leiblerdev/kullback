# Complete source of three small tool-use environments (tau2-bench mock, OpenEnv echo_env, verifiers wiki_search)

Source: research agent, 2026-08-26, fetched verbatim from GitHub main branches. Upstream code is quoted exactly (including any dashes inside it).

All source is in hand. Below is the complete package. One note on scope before the code: the verbatim requirement won over the 1,200 line target. The total runs roughly 1,700 lines because tau2's replay-based verifier (`set_state`, `EnvironmentEvaluator`, `ActionEvaluator`) is itself about 450 lines and cannot be cut without elisions. What I left out entirely rather than elide: the `FullDuplex*` evaluator classes (they duplicate the half-duplex ones for tick trajectories), the non-`ALL` branches of `evaluate_simulation`, echo_env's `Dockerfile`/`pyproject.toml`/`uv.lock`, and the verifiers `pyproject.toml`. Two things to know up front:

- `meta-pytorch/OpenEnv` now redirects to `huggingface/OpenEnv` (GitHub API returns 301). Files below are from the `main` branch there.
- echo_env has no `models.py` even though its README's "Project Structure" lists one. It is a pure MCP environment and reuses `CallToolAction` / `CallToolObservation` from `openenv/core/env_server/mcp_types.py`, which I include instead.
- A handful of em dashes appear inside quoted upstream code and docstrings. They are preserved because the code is quoted exactly as fetched; none appear in my own prose.

---

# 1. tau2-bench "mock" domain (Sierra)

Repo: `sierra-research/tau2-bench`, branch `main`. Layout: code in `src/tau2/domains/mock/`, data in `data/tau2/domains/mock/`, verifier in `src/tau2/evaluator/`.

## 1.1 `src/tau2/domains/mock/utils.py`

```python
from tau2.utils.utils import DATA_DIR

MOCK_DATA_DIR = DATA_DIR / "tau2" / "domains" / "mock"
MOCK_DB_PATH = MOCK_DATA_DIR / "db.json"
MOCK_USER_DB_PATH = MOCK_DATA_DIR / "user_db.json"
MOCK_POLICY_PATH = MOCK_DATA_DIR / "policy.md"
MOCK_POLICY_SOLO_PATH = MOCK_DATA_DIR / "policy_solo.md"
MOCK_TASK_SET_PATH = MOCK_DATA_DIR / "tasks.json"
```

## 1.2 `src/tau2/domains/mock/data_model.py` (agent-side DB)

```python
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from tau2.domains.mock.utils import MOCK_DB_PATH
from tau2.environment.db import DB

TaskStatus = Literal["pending", "completed"]


class Task(BaseModel):
    task_id: str = Field(description="Unique identifier for the task")
    title: str = Field(description="Title of the task")
    description: Optional[str] = Field(None, description="Description of the task")
    status: TaskStatus = Field(description="Status of the task")


class User(BaseModel):
    user_id: str = Field(description="Unique identifier for the user")
    name: str = Field(description="User's name")
    tasks: List[str] = Field(description="List of task IDs assigned to the user")


class MockDB(DB):
    """Simple database with users and their tasks."""

    tasks: Dict[str, Task] = Field(
        description="Dictionary of all tasks indexed by task ID"
    )
    users: Dict[str, User] = Field(
        description="Dictionary of all users indexed by user ID"
    )


def get_db():
    return MockDB.load(MOCK_DB_PATH)
```

## 1.3 `src/tau2/environment/db.py` (base class every domain DB inherits)

```python
from typing import Any, Optional

from tau2.utils import dump_file, get_pydantic_hash, load_file
from tau2.utils.pydantic_utils import BaseModelNoExtra


class DB(BaseModelNoExtra):
    """Domain database.

    This is a base class for all domain databases.
    """

    @classmethod
    def load(cls, path: str) -> "DB":
        """Load the database from a structured file like JSON, YAML, or TOML."""
        data = load_file(path)
        return cls.model_validate(data)

    def dump(self, path: str, exclude_defaults: bool = False, **kwargs: Any) -> None:
        """Dump the database to a file."""
        data = self.model_dump(exclude_defaults=exclude_defaults)
        dump_file(path, data, **kwargs)

    def get_json_schema(self) -> dict[str, Any]:
        """Get the JSON schema of the database."""
        return self.model_json_schema()

    def get_hash(self) -> str:
        """Get the hash of the database."""
        return get_pydantic_hash(self)

    def get_statistics(self) -> dict[str, Any]:
        """Get the statistics of the database."""
        return {}


def get_db_json_schema(db: Optional[DB] = None) -> dict[str, Any]:
    """Get the JSONschema of the database."""
    if db is None:
        return {}
    return db.get_json_schema()
```

## 1.4 `src/tau2/domains/mock/tools.py` (agent-side tools)

```python
from tau2.domains.mock.data_model import MockDB, Task, TaskStatus, User
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class MockTools(ToolKitBase):
    """Simple tools for the mock domain."""

    db: MockDB

    def __init__(self, db: MockDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.WRITE)
    def create_task(self, user_id: str, title: str, description: str = None) -> Task:
        """
        Create a new task for a user.

        Args:
            user_id: The ID of the user creating the task
            title: The title of the task
            description: Optional description of the task

        Returns:
            The created task

        Raises:
            ValueError: If the user is not found
        """
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")

        task_id = f"task_{len(self.db.tasks) + 1}"
        task = Task(
            task_id=task_id, title=title, description=description, status="pending"
        )

        self.db.tasks[task_id] = task
        self.db.users[user_id].tasks.append(task_id)

        return task

    @is_tool(ToolType.READ)
    def get_users(self) -> list[User]:
        """
        Get all users in the database.
        """
        return list(self.db.users.values())

    @is_tool(ToolType.WRITE)
    def update_task_status(self, task_id: str, status: TaskStatus) -> Task:
        """
        Update the status of a task.

        Args:
            task_id: The ID of the task to update
            status: The new status of the task

        Returns:
            The updated task

        Raises:
            ValueError: If the task is not found
        """
        if task_id not in self.db.tasks:
            raise ValueError(f"Task {task_id} not found")

        task = self.db.tasks[task_id]
        task.status = status
        return task

    def assert_number_of_tasks(self, user_id: str, expected_number: int) -> bool:
        """
        Check if the number of tasks for a user is as expected.

        Args:
            user_id: The ID of the user
            expected_number: The expected number of tasks

        Returns:
            True if the number of tasks is as expected, False otherwise
        """
        if user_id not in self.db.users:
            raise ValueError(f"User {user_id} not found")
        return len(self.db.users[user_id].tasks) == expected_number

    def assert_task_status(self, task_id: str, expected_status: TaskStatus) -> bool:
        """
        Check if the status of a task is as expected.
        """
        if task_id not in self.db.tasks:
            raise ValueError(f"Task {task_id} not found")
        return self.db.tasks[task_id].status == expected_status

    @is_tool(ToolType.GENERIC)
    def transfer_to_human_agents(self, summary: str) -> str:
        """
        Transfer the user to a human agent, with a summary of the user's issue.
        Only transfer if
         -  the user explicitly asks for a human agent
         -  given the policy and the available tools, you cannot solve the user's issue.

        Args:
            summary: A summary of the user's issue.

        Returns:
            A message indicating the user has been transferred to a human agent.
        """
        return "Transfer successful"

    # @is_tool(ToolType.THINK)
    # def think(self, thought: str) -> str:
    #     """
    #     Use the tool to think about something.
    #     It will not obtain new information or change the database, but just append the thought to the log.
    #     Use it when complex reasoning or some cache memory is needed.

    #     Args:
    #         thought: A thought to think about.

    #     Returns:
    #         Empty string
    #     """
    #     return ""
```

## 1.5 `src/tau2/domains/mock/user_data_model.py` (user-side DB)

```python
from typing import Dict, Literal, Optional

from pydantic import Field

from tau2.environment.db import DB
from tau2.utils.pydantic_utils import BaseModelNoExtra

NotificationStatus = Literal["unread", "read"]


class Notification(BaseModelNoExtra):
    """A notification in the user's inbox."""

    notification_id: str = Field(description="Unique identifier for the notification")
    message: str = Field(description="The notification message")
    status: NotificationStatus = Field(
        default="unread", description="Whether the notification has been read"
    )
    task_id: Optional[str] = Field(
        None, description="Associated task ID, if applicable"
    )


class MockUserDB(DB):
    """Simple user database with a notification inbox."""

    notifications: Dict[str, Notification] = Field(
        default_factory=dict,
        description="Dictionary of notifications indexed by notification ID",
    )
```

## 1.6 `src/tau2/domains/mock/user_tools.py` (tools the simulated user can call)

```python
from tau2.domains.mock.user_data_model import MockUserDB, Notification
from tau2.environment.toolkit import ToolKitBase, ToolType, is_tool


class MockUserTools(ToolKitBase):
    """User-side tools for the mock domain.

    Simulates a notification inbox where the user can check and manage
    notifications about their tasks.
    """

    db: MockUserDB

    def __init__(self, db: MockUserDB) -> None:
        super().__init__(db)

    @is_tool(ToolType.READ)
    def check_notifications(self) -> list[Notification]:
        """Check all notifications in the user's inbox.

        Returns:
            A list of all notifications.
        """
        return list(self.db.notifications.values())

    @is_tool(ToolType.WRITE)
    def dismiss_notification(self, notification_id: str) -> str:
        """Dismiss (mark as read) a notification.

        Args:
            notification_id: The ID of the notification to dismiss.

        Returns:
            A confirmation message.

        Raises:
            ValueError: If the notification is not found.
        """
        if notification_id not in self.db.notifications:
            raise ValueError(f"Notification {notification_id} not found")
        self.db.notifications[notification_id].status = "read"
        return f"Notification {notification_id} dismissed"

    # --- Non-tool helpers (for initialization_actions / env_assertions) ---

    def add_notification(
        self,
        notification_id: str,
        message: str,
        task_id: str | None = None,
    ) -> Notification:
        """Add a notification to the user's inbox (used by initialization_actions)."""
        notification = Notification(
            notification_id=notification_id,
            message=message,
            task_id=task_id,
        )
        self.db.notifications[notification_id] = notification
        return notification

    def assert_notification_status(
        self, notification_id: str, expected_status: str
    ) -> bool:
        """Check if a notification has the expected status (used by env_assertions)."""
        if notification_id not in self.db.notifications:
            raise ValueError(f"Notification {notification_id} not found")
        return self.db.notifications[notification_id].status == expected_status
```

## 1.7 `src/tau2/domains/mock/environment.py` (wiring)

```python
from pathlib import Path
from typing import Optional

from tau2.data_model.tasks import Task
from tau2.domains.mock.data_model import MockDB
from tau2.domains.mock.tools import MockTools
from tau2.domains.mock.user_data_model import MockUserDB
from tau2.domains.mock.user_tools import MockUserTools
from tau2.domains.mock.utils import (
    MOCK_DB_PATH,
    MOCK_POLICY_PATH,
    MOCK_POLICY_SOLO_PATH,
    MOCK_TASK_SET_PATH,
    MOCK_USER_DB_PATH,
)
from tau2.environment.environment import Environment
from tau2.utils import load_file


def get_environment(
    db: Optional[MockDB] = None,
    user_db: Optional[MockUserDB] = None,
    solo_mode: bool = False,
) -> Environment:
    if db is None:
        db = MockDB.load(MOCK_DB_PATH)
    tools = MockTools(db)
    if user_db is None:
        user_db = MockUserDB.load(MOCK_USER_DB_PATH)
    user_tools = MockUserTools(user_db)
    if not solo_mode:
        policy_path = MOCK_POLICY_PATH
    else:
        policy_path = MOCK_POLICY_SOLO_PATH
    with open(policy_path, "r") as fp:
        policy = fp.read()
    env = Environment(
        domain_name="mock",
        policy=policy,
        tools=tools,
        user_tools=user_tools,
    )
    if solo_mode:
        env.set_solo_mode(True)
    return env


# def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
#     if task_split_name is not None:
#         raise ValueError(f"Task split not supported for mock domain")
#     with open(MOCK_TASK_SET_PATH, "r") as fp:
#         tasks = json.load(fp)
#     return [Task.model_validate(task) for task in tasks]


def get_tasks(task_split_name: Optional[str] = None) -> list[Task]:
    tasks = load_file(MOCK_TASK_SET_PATH)
    tasks = [Task.model_validate(task) for task in tasks]
    if task_split_name is None:
        return tasks
    task_splits = get_tasks_split()
    if task_split_name not in task_splits:
        raise ValueError(
            f"Invalid task split name: {task_split_name}. Valid splits are: {task_splits.keys()}"
        )
    return [task for task in tasks if task.id in task_splits[task_split_name]]


def get_tasks_split() -> dict[str, list[str]]:
    split_file = (
        Path(MOCK_TASK_SET_PATH).parent / f"split_{Path(MOCK_TASK_SET_PATH).stem}.json"
    )
    return load_file(split_file)
```

## 1.8 Data files

`data/tau2/domains/mock/db.json`

```json
{
  "tasks": {
    "task_1": {
      "task_id": "task_1",
      "title": "Test task",
      "description": "A test task",
      "status": "pending"
    }
  },
  "users": {
    "user_1": {
      "user_id": "user_1", 
      "name": "Test User",
      "tasks": ["task_1"]
    }
  }
} 
```

`data/tau2/domains/mock/user_db.json`

```json
{
  "notifications": {
    "notif_1": {
      "notification_id": "notif_1",
      "message": "Task 'Test task' has been assigned to you.",
      "status": "unread",
      "task_id": "task_1"
    }
  }
}
```

`data/tau2/domains/mock/policy.md`

```markdown
# Mock Domain Policy

1. Each task must have a title
2. Task status can only be "pending" or "completed"
3. Only existing users can create tasks 
4. You are not allowed to delete tasks. You should transfer the a human agent.
5. If the user asks for a compliment, compliment them
```

`data/tau2/domains/mock/split_tasks.json`

```json
{
    "base": [
        "create_task_1",
        "create_task_1_with_env_assertions",
        "create_task_1_nl_eval",
        "update_task_1",
        "update_task_with_message_history",
        "update_task_with_initialization_data",
        "update_task_with_initialization_actions",
        "update_task_with_history_and_env_assertions",
        "update_task_with_user_tools",
        "impossible_task_1"
    ]
}
```

## 1.9 Four tasks from `data/tau2/domains/mock/tasks.json` (10 tasks total in the file)

These four cover every evaluation field: `actions`, `env_assertions`, `nl_assertions`, `communicate_info`, `reward_basis`, `compare_args`, and both `initialization_data` and `initialization_actions`.

```json
  {
    "id": "create_task_1_with_env_assertions",
    "description": {
      "purpose": "Test the create_task functionality",
      "notes": "Basic task creation test with a simple title"
    },
    "user_scenario": {
      "persona": "Professional and direct communicator who writes in a clear and concise style",
      "instructions": "You are a team member who needs to create a task for an upcoming meeting. Your goal is to create a task to track this important meeting. Create a new task called 'Important Meeting' for user_1."
    },
    "ticket": "User needs to create a task for an upcoming meeting. Create a new task called 'Important Meeting' for user_1.",
    "evaluation_criteria": {
      "actions": [
        {
          "action_id": "create_1",
          "name": "create_task",
          "arguments": {
            "user_id": "user_1",
            "title": "Important Meeting"
          },
          "info": "Create a new task for the meeting"
        }
      ],
      "env_assertions": [
        {
          "env_type": "assistant",
          "func_name": "assert_task_status",
          "arguments": {"task_id": "task_2", "expected_status": "pending"}
        }
      ],
      "nl_assertions": [
        "The agent confirmed the task was created successfully"
      ], 
      "reward_basis": ["DB", "ENV_ASSERTION"]
    }
  },
  {
    "id": "update_task_with_initialization_data",
    "description": {
      "purpose": "Test the update_task_status functionality with pre-existing conversation history",
      "notes": "Tests updating a task status with initial message history showing previous context"
    },
    "user_scenario": {
      "persona": "Professional and direct communicator",
      "instructions": "Continue the conversation about task management. The previous discussion was about creating a task with ID task_2. Now you want to mark the task as completed."
    },
    "ticket": "User needs to update the status of task_2 to completed.",
    "initial_state": {
      "initialization_data": {
        "agent_data": {
          "tasks": {
            "task_2": {
              "task_id": "task_2",
              "title": "Project Review",
              "description": "Review Q4 project status",
              "status": "pending"
            }
          },
          "users": {
            "user_1": {
              "tasks": ["task_1", "task_2"]
            }
          }
        }
      }
    },
    "evaluation_criteria": {
      "actions": [
        {
          "action_id": "update_1",
          "name": "update_task_status",
          "arguments": {
            "task_id": "task_2",
            "status": "completed"
          },
          "info": "Update the task status to completed"
        }
      ],
      "communicate_info": [
        "The agent acknowledged the previous context",
        "The agent confirmed the task status was updated successfully"
      ]
    }
  },
  {
    "id": "update_task_with_user_tools",
    "description": {
      "purpose": "Test user tool calling with check_notifications and dismiss_notification",
      "notes": "The user checks their notification inbox, sees a task assigned to them, asks the agent to mark it completed, then dismisses the notification."
    },
    "user_scenario": {
      "persona": "Professional and direct communicator who writes in a clear and concise style",
      "instructions": "You just received a notification about a task. First, use the check_notifications tool to see your notifications. You will see that task_1 ('Test task') has been assigned to you. Tell the agent you have completed this task and ask them to update the status to completed. After the agent confirms, use dismiss_notification to dismiss the notification 'notif_1'."
    },
    "ticket": "User checks their notifications, sees task_1 was assigned, and asks the agent to mark task_1 as completed.",
    "initial_state": {
      "initialization_actions": [
        {
          "env_type": "user",
          "func_name": "add_notification",
          "arguments": {
            "notification_id": "notif_1",
            "message": "Task 'Test task' has been assigned to you.",
            "task_id": "task_1"
          }
        }
      ]
    },
    "evaluation_criteria": {
      "actions": [
        {
          "action_id": "update_1",
          "name": "update_task_status",
          "arguments": {
            "task_id": "task_1",
            "status": "completed"
          },
          "info": "Update the task status to completed"
        }
      ],
      "env_assertions": [
        {
          "env_type": "user",
          "func_name": "assert_notification_status",
          "arguments": {"notification_id": "notif_1", "expected_status": "read"},
          "assert_value": true
        }
      ],
      "nl_assertions": [
        "The agent confirmed the task status was updated successfully"
      ],
      "reward_basis": ["DB", "ACTION", "ENV_ASSERTION"]
    }
  },
  {
    "id": "impossible_task_1",
    "description": {
      "purpose": "Test delete_task functionality",
      "notes": "Asks the agent to delete a task."
    },
    "user_scenario": {
      "persona": "Professional and direct communicator who writes in a clear and concise style",
      "instructions": "You want to delete all your current tasks."
    },
    "ticket": "User needs to delete all their current tasks.",
    "evaluation_criteria": {
      "actions": [
        {
          "action_id": "transfer_1",
          "name": "transfer_to_human_agents",
          "arguments": {
            "summary": "User needs to delete all their current tasks. This is not possible to do with the tools available."
          },
          "compare_args": [],
          "info": "Transfer the user to a human agent"
        }
      ], 
      "reward_basis": ["DB", "ACTION"]
    }
  }
```

## 1.10 How reward is computed

### `src/tau2/data_model/tasks.py`, `RewardType` and `Action.compare_with_tool_call` (two excerpts from a 685-line file)

```python
class RewardType(str, Enum):
    """
    Components that can gate a task's final reward.

    The final reward is the product of every component listed in
    `EvaluationCriteria.reward_basis`. Components not listed are not
    included (they may still run for diagnostics). Default basis is
    `[DB, COMMUNICATE]`, matching the original τ-bench.

    - DB: predicted DB end state matches the target. Target is the
      result of replaying `EvaluationCriteria.actions` on a fresh env;
      any agent path producing an equivalent end state passes.
    - ENV_ASSERTION: all `env_assertions` pass on the predicted env.
    - COMMUNICATE: every `communicate_info` string appears (substring)
      in the agent's messages.
    - NL_ASSERTION: every `nl_assertions` entry is judged true by an
      LLM (experimental / WIP).
    - ACTION: every entry in `actions` is matched by an agent tool call
      (per `Action.compare_with_tool_call`). The only reward type that
      makes the action list a hard requirement — promotes it to the
      assumed-unique correct trajectory. Used in a few
      `banking_knowledge` tasks; not used in airline/retail/telecom.

    See `docs/evaluation.md`.
    """

    DB = "DB"
    ENV_ASSERTION = "ENV_ASSERTION"
    NL_ASSERTION = "NL_ASSERTION"
    ACTION = "ACTION"
    COMMUNICATE = "COMMUNICATE"
```

```python
    def compare_with_tool_call(self, tool_call: ToolCall) -> bool:
        """
        Compare the action with a tool call.
        If the name is not the same, return False.
        If compare_args is None, will check all the arguments.
        Otherwise, will check only the arguments in compare_args.
        """
        if self.name != tool_call.name:
            return False
        if self.compare_args is None:
            compare_args = tool_call.arguments.keys()
        else:
            compare_args = self.compare_args
        if len(compare_args) == 0:
            return True
        tool_args = {k: v for k, v in tool_call.arguments.items() if k in compare_args}
        action_args = {k: v for k, v in self.arguments.items() if k in compare_args}
        return tool_args == action_args
```

### `src/tau2/environment/environment.py`, `set_state` and `run_env_assertion` (two methods from a 490-line file)

This is the reset-and-replay mechanism. A fresh `Environment` is built from `db.json`, then `set_state` applies `initialization_data` (a partial DB overlay), `initialization_actions` (function calls on the toolkits), and finally replays every mutating tool call from the message history against the live tools, checking each output against the recorded `ToolMessage`.

```python
    def set_state(
        self,
        initialization_data: Optional[InitializationData],
        initialization_actions: Optional[list[EnvFunctionCall]],
        message_history: list[Message],
        strict: bool = True,
    ):
        """
        Set the state of the environment given initialization data and a list of messages.

        Args:
            strict: When True (default), raise if a replayed mutating tool call
                returns different content than the recorded ToolMessage. When
                False, log a warning instead and continue the replay. Lenient
                mode is intended for re-grading historical trajectories whose
                recorded tool outputs contain cosmetic drift against current
                tool code (e.g. numeric argument echoes rendered as `25` by the
                code that produced them but `25.0` after numeric-argument
                normalization); the state mutation is applied identically
                either way.
        """
        if self.solo_mode:
            assert all(
                [not isinstance(message, UserMessage) for message in message_history]
            ), "User messages are not allowed in solo mode"

        def get_actions_from_messages(
            messages: list[Message],
        ) -> list[tuple[ToolCall, ToolMessage]]:
            """
            Get the actions from the messages.
            """
            messages = deepcopy(messages)[::-1]
            actions = []
            while messages:
                message = messages.pop()
                if isinstance(message, ToolMessage):
                    raise ValueError(
                        "Tool message not expected. Tool messages should always follow a tool call."
                    )
                if (
                    isinstance(message, (AssistantMessage, UserMessage))
                    and message.is_tool_call()
                ):
                    tool_calls = message.tool_calls
                    for tc in tool_calls:
                        if len(messages) == 0:
                            raise ValueError("Tool message expected. Got None.")
                        tm = messages.pop()
                        if not isinstance(tm, ToolMessage):
                            raise ValueError(f"Tool message expected. Got {type(tm)}")
                        if tc.id != tm.id:
                            raise ValueError(
                                f"Tool call id mismatch. Got {tc.id} and {tm.id}"
                            )
                        actions.append((tc, tm))

            return actions

        if initialization_data is not None:
            if initialization_data.agent_data is not None:
                self.tools.update_db(initialization_data.agent_data)
                # Sync user_tools.db to point to the same db instance as tools.db
                # This is necessary because update_db creates a new db instance
                if self.user_tools is not None and self.user_tools.db is not None:
                    self.user_tools.db = self.tools.db
            if initialization_data.user_data is not None:
                self.user_tools.update_db(initialization_data.user_data)
                # Sync tools.db to point to the same db instance as user_tools.db
                if self.tools is not None and self.tools.db is not None:
                    self.tools.db = self.user_tools.db

        if initialization_actions is not None:
            for action in initialization_actions:
                self.run_env_function_call(action)

        action_responses = get_actions_from_messages(message_history)
        for tool_call, expected_response in action_responses:
            if not self._has_tool(tool_call.name):
                # Hallucinated tool name. The live env returned a
                # ToolMessage(error=True) for this call and made no state
                # change, so replay it as a no-op. The agent's subsequent
                # recovery (if any) will still be replayed and determine
                # the final state. Repeated hallucination is bounded
                # upstream by the orchestrator's max_errors guard, which
                # ends the live sim with TerminationReason.TOO_MANY_ERRORS
                # before evaluation runs.
                logger.debug(
                    f"Skipping unknown tool '{tool_call.name}' during replay "
                    "(no-op, matching live env behavior on hallucinated tools)."
                )
                continue
            # Non-mutating tools (reads, thinks, etc.) don't change state --
            # skip them to avoid re-execution and non-deterministic output
            # comparison issues.
            if not self._is_mutating_tool(tool_call.name):
                continue
            response = self.get_response(tool_call)
            try:
                content = json.loads(response.content)
            except json.JSONDecodeError:
                content = response.content
            try:
                expected_content = json.loads(expected_response.content)
            except json.JSONDecodeError:
                expected_content = expected_response.content
            if content != expected_content:
                if strict:
                    raise ValueError(
                        f"Tool call:\n{tool_call}\n\nReturned:\n{response}\n\nExpected:\n{expected_response}"
                    )
                logger.warning(
                    f"Replayed tool call '{tool_call.name}' returned different "
                    f"content than the recorded ToolMessage; continuing because "
                    f"strict=False. Recorded output may predate current tool "
                    f"code.\nTool call:\n{tool_call}"
                )
        self.sync_tools()

    def run_env_assertion(
        self,
        assertion: EnvAssertion,
        raise_assertion_error: bool = True,
    ) -> bool:
        """
        Runs any assertion function on agent tools or user tools.
        """
        if not isinstance(assertion, EnvAssertion):
            raise ValueError(f"Assertion must be an EnvAssertion. Got {assertion}")
        res = self.run_env_function_call(assertion)
        if not isinstance(res, bool):
            raise ValueError(
                f"Function {assertion.func_name} returned {type(res)} instead of bool"
            )
        assert_pass = res == assertion.assert_value
        if raise_assertion_error:
            assert assert_pass, assertion.message or f"Assertion failed: {assertion}"
        return assert_pass
```

### `src/tau2/evaluator/evaluator_env.py`, DB check (the `EnvironmentEvaluator` class, complete; the `FullDuplexEnvironmentEvaluator` that follows it in the file is omitted)

```python
from typing import Callable

from loguru import logger

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    UserMessage,
)
from tau2.data_model.simulation import DBCheck, EnvAssertionCheck, RewardInfo
from tau2.data_model.tasks import RewardType, Task
from tau2.environment.environment import Environment
from tau2.evaluator.evaluator_base import EvaluatorBase


class EnvironmentEvaluator(EvaluatorBase[Message]):
    """
    Evaluator focuses on endstate of the simulation environment.
    """

    @classmethod
    def calculate_reward(
        cls,
        environment_constructor: Callable[[], Environment],
        task: Task,
        full_trajectory: list[
            Message
        ],  # FIXME: It would be better to be able to get only the messages that are after the initial state
        solo_mode: bool = False,
        env_kwargs: dict = None,
        strict_replay: bool = True,
    ) -> RewardInfo:
        """
        Calculate the reward for the simulation.
        Args:
            environment_constructor: Callable[[], Environment]
            task: Task
            full_trajectory: list[Message] (Must include the message history from task initial state)
            solo_mode: bool
            strict_replay: forwarded to Environment.set_state(strict=...). Set
                False when re-grading historical trajectories whose recorded
                tool outputs may cosmetically differ from current tool code.
        Returns:
            RewardInfo
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                info={"note": "No evaluation criteria"},
            )
        expected_actions = task.evaluation_criteria.actions
        env_assertions = task.evaluation_criteria.env_assertions
        if expected_actions is None and env_assertions is None:
            return RewardInfo(
                reward=1.0,
                db_check=DBCheck(db_match=True, db_reward=1.0),
                info={"note": "No expected actions or env assertions"},
            )

        initialization_data = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_data is not None
        ):
            initialization_data = task.initial_state.initialization_data

        initialization_actions = None
        if (
            task.initial_state is not None
            and task.initial_state.initialization_actions is not None
        ):
            initialization_actions = task.initial_state.initialization_actions

        message_history = []
        if (
            task.initial_state is not None
            and task.initial_state.message_history is not None
        ):
            message_history = task.initial_state.message_history

        if env_kwargs is None:
            env_kwargs = {}

        predicted_environment = environment_constructor(
            solo_mode=solo_mode, **env_kwargs
        )

        predicted_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=list(full_trajectory),
            strict=strict_replay,
        )

        # Setting up gold environment
        gold_environment = environment_constructor(**env_kwargs)
        gold_environment.set_state(
            initialization_data=initialization_data,
            initialization_actions=initialization_actions,
            message_history=message_history,
            strict=strict_replay,
        )
        golden_actions = task.evaluation_criteria.actions or []
        for action in golden_actions:
            try:
                gold_environment.make_tool_call(
                    tool_name=action.name,
                    requestor=action.requestor,
                    **action.arguments,
                )
            except Exception as e:
                logger.warning(
                    f"Error in golden actions {action.name}({action.arguments}): {e}"
                )

        # Comparing the environments
        agent_db_hash = gold_environment.get_db_hash()
        user_db_hash = gold_environment.get_user_db_hash()
        predicted_agent_db_hash = predicted_environment.get_db_hash()
        predicted_user_db_hash = predicted_environment.get_user_db_hash()
        agent_db_match = agent_db_hash == predicted_agent_db_hash
        user_db_match = user_db_hash == predicted_user_db_hash
        if agent_db_match and user_db_match:
            db_reward = 1.0
            db_match = True
        else:
            db_reward = 0.0
            db_match = False

        db_check = DBCheck(db_match=db_match, db_reward=db_reward)

        # Run env assertions
        env_assertions = task.evaluation_criteria.env_assertions or []
        env_assertion_checks = []
        env_assertion_reward = 1.0
        for env_assertion in env_assertions:
            success = predicted_environment.run_env_assertion(
                env_assertion,
                raise_assertion_error=False,
            )
            res = EnvAssertionCheck(
                env_assertion=env_assertion,
                met=success,
                reward=1.0 if success else 0.0,
            )
            env_assertion_checks.append(res)
            env_assertion_reward *= res.reward

        reward = 1.0
        reward_breakdown = {}
        if RewardType.DB in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.DB] = db_reward
            reward *= db_reward
        if RewardType.ENV_ASSERTION in task.evaluation_criteria.reward_basis:
            reward_breakdown[RewardType.ENV_ASSERTION] = env_assertion_reward
            reward *= env_assertion_reward

        return RewardInfo(
            reward=reward,
            db_check=db_check,
            env_assertions=env_assertion_checks,
            reward_basis=task.evaluation_criteria.reward_basis,
            reward_breakdown=reward_breakdown,
        )
```

### `src/tau2/evaluator/evaluator_action.py`, action check (imports, `_check_actions`, and `ActionEvaluator`, complete; `FullDuplexActionEvaluator` omitted)

```python
from typing import Optional

from tau2.data_model.message import (
    AssistantMessage,
    Message,
    Tick,
    ToolCall,
    UserMessage,
)
from tau2.data_model.simulation import ActionCheck, RewardInfo
from tau2.data_model.tasks import Action, RewardType, Task
from tau2.environment.toolkit import ToolType
from tau2.evaluator.evaluator_base import EvaluatorBase


def _check_actions(
    predicted_tool_calls: list[ToolCall],
    golden_actions: list[Action],
    tool_types: Optional[dict[str, ToolType]] = None,
) -> list[ActionCheck]:
    """
    Check if all the gold actions are in the predicted tool calls.

    Args:
        predicted_tool_calls: List of tool calls made during the simulation.
        golden_actions: List of expected actions from evaluation criteria.
        tool_types: Optional mapping of tool names to their types (read/write/think/generic).

    Returns:
        List of ActionCheck results for each golden action.
    """
    action_checks = []
    for gold_action in golden_actions:
        found = False
        for pred_tool_call in predicted_tool_calls:
            if gold_action.compare_with_tool_call(pred_tool_call):
                found = True
                break
        if found:
            gold_action_reward = 1.0
            gold_action_match = True
        else:
            gold_action_reward = 0.0
            gold_action_match = False

        # Get tool type if available
        tool_type = None
        if tool_types is not None:
            tool_type = tool_types.get(gold_action.name)

        action_checks.append(
            ActionCheck(
                action=gold_action,
                action_match=gold_action_match,
                action_reward=gold_action_reward,
                tool_type=tool_type,
            )
        )
    return action_checks


class ActionEvaluator(EvaluatorBase[Message]):
    """
    Evaluates whether or not the agent performed the required actions.
    """

    @classmethod
    def calculate_reward(
        cls,
        task: Task,
        full_trajectory: list[Message],
        tool_types: Optional[dict[str, ToolType]] = None,
    ) -> RewardInfo:
        """
        Calculate the reward based on whether the agent performed the required actions.

        Args:
            task: The task containing evaluation criteria.
            full_trajectory: List of messages from the simulation.
            tool_types: Optional mapping of tool names to their types (read/write/think/generic).
        """
        if task.evaluation_criteria is None:
            return RewardInfo(
                reward=1.0,
                action_checks=[],
                info={"note": "No evaluation criteria"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )
        golden_actions = task.evaluation_criteria.actions
        if not golden_actions:
            return RewardInfo(
                reward=1.0,
                info={"note": "No actions to evaluate"},
                reward_breakdown={RewardType.ACTION: 1.0},
            )

        action_checks = cls.evaluate_actions(
            full_trajectory, golden_actions, tool_types
        )

        # Calculate reward: 1 if all expectations are met, 0 otherwise
        all_expectations_met = all(result.action_match for result in action_checks)
        reward = 1.0 if all_expectations_met else 0.0

        return RewardInfo(
            reward=reward,
            action_checks=action_checks,
            reward_breakdown={RewardType.ACTION: reward},
        )

    @classmethod
    def extract_tool_calls(cls, full_trajectory: list[Message]) -> list[ToolCall]:
        """
        Extract all tool calls from a message trajectory.

        Args:
            full_trajectory: List of messages from the simulation.

        Returns:
            List of ToolCall objects extracted from AssistantMessage and UserMessage.
        """
        tool_calls: list[ToolCall] = []
        for message in full_trajectory:
            if (
                isinstance(message, AssistantMessage)
                or isinstance(message, UserMessage)
            ) and message.is_tool_call():
                tool_calls.extend(message.tool_calls)
        return tool_calls

    @classmethod
    def evaluate_actions(
        cls,
        full_trajectory: list[Message],
        golden_actions: list[Action],
        tool_types: Optional[dict[str, ToolType]] = None,
    ) -> list[ActionCheck]:
        """
        Evaluate whether the agent performed the expected actions.

        Args:
            full_trajectory: List of messages from the simulation.
            golden_actions: List of expected actions from evaluation criteria.
            tool_types: Optional mapping of tool names to their types (read/write/think/generic).
        """
        if len(golden_actions) == 0:
            return []

        predicted_tool_calls = cls.extract_tool_calls(full_trajectory)
        return _check_actions(predicted_tool_calls, golden_actions, tool_types)
```

### `src/tau2/evaluator/evaluator.py`, the `ALL` branch of `evaluate_simulation` (excerpt: the function's early exits, the `ENV`/`ACTION`/`COMMUNICATE`/`NL_ASSERTIONS`-only branches, and the `*_IGNORE_BASIS` branches are omitted)

```python
    elif evaluation_type in {EvaluationType.ALL, EvaluationType.ALL_WITH_NL_ASSERTIONS}:
        env_reward_info = EnvEvaluator.calculate_reward(
            environment_constructor=registry.get_env_constructor(domain),
            task=task,
            full_trajectory=trajectory,
            solo_mode=solo_mode,
            env_kwargs=env_kwargs,
            strict_replay=strict_replay,
        )
        action_reward_info = ActEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory,
            tool_types=tool_types,
        )
        communicate_reward_info = CommEvaluator.calculate_reward(
            task=task,
            full_trajectory=trajectory,
        )
        nl_reward_info = None
        task_needs_nl = RewardType.NL_ASSERTION in task.evaluation_criteria.reward_basis
        if evaluation_type == EvaluationType.ALL_WITH_NL_ASSERTIONS or task_needs_nl:
            nl_reward_info = NLEvaluator.calculate_reward(
                task=task,
                full_trajectory=trajectory,
            )

        ## Combine all the rewards.
        reward = 1.0
        env_bases = {RewardType.DB, RewardType.ENV_ASSERTION}
        action_bases = {RewardType.ACTION}
        nl_bases = {RewardType.NL_ASSERTION}
        comm_bases = {RewardType.COMMUNICATE}
        task_reward_basis = set(task.evaluation_criteria.reward_basis)

        evaluated_bases = env_bases | action_bases | comm_bases
        if nl_reward_info is not None:
            evaluated_bases |= nl_bases
        unevaluated = task_reward_basis - evaluated_bases
        if unevaluated:
            raise ValueError(
                f"Task reward_basis includes {unevaluated} but these were "
                f"not evaluated. evaluation_type={evaluation_type.value}"
            )

        reward_breakdown = {}
        if task_reward_basis & env_bases:
            if env_reward_info.reward_breakdown is not None:
                reward_breakdown.update(env_reward_info.reward_breakdown)
            reward *= env_reward_info.reward
        if task_reward_basis & action_bases:
            if action_reward_info.reward_breakdown is not None:
                reward_breakdown.update(action_reward_info.reward_breakdown)
            reward *= action_reward_info.reward
        if task_reward_basis & nl_bases:
            if nl_reward_info.reward_breakdown is not None:
                reward_breakdown.update(nl_reward_info.reward_breakdown)
            reward *= nl_reward_info.reward
        if task_reward_basis & comm_bases:
            if communicate_reward_info.reward_breakdown is not None:
                reward_breakdown.update(communicate_reward_info.reward_breakdown)
            reward *= communicate_reward_info.reward

        reward_info = RewardInfo(
            reward=reward,
            db_check=env_reward_info.db_check,
            env_assertions=env_reward_info.env_assertions,
            action_checks=action_reward_info.action_checks,
            nl_assertions=(
                nl_reward_info.nl_assertions if nl_reward_info is not None else None
            ),
            communicate_checks=communicate_reward_info.communicate_checks,
            reward_basis=task.evaluation_criteria.reward_basis,
            reward_breakdown=reward_breakdown,
            info={
                "env": env_reward_info.info,
                "nl": nl_reward_info.info if nl_reward_info is not None else None,
                "communicate": communicate_reward_info.info,
                "action": action_reward_info.info,
            },
        )
```

Also worth knowing from the omitted top of `evaluate_simulation`: if `simulation.termination_reason` is not `AGENT_STOP` or `USER_STOP`, reward is 0.0 immediately; if `task.evaluation_criteria is None`, reward is 1.0.

## 1.11 Commentary: tau2 mock

1. State is a Pydantic model (`MockDB`) holding two dicts, `tasks` and `users`, plus a second Pydantic model (`MockUserDB`) for the simulated user's notification inbox. Both inherit `DB`, which adds `load(path)` and `get_hash()`. There is no database engine; the "DB" is an in-memory object tree.
2. Tools are plain methods on a `ToolKitBase` subclass that holds `self.db`. `@is_tool(ToolType.WRITE|READ|GENERIC|THINK)` marks which methods are exposed to the model and whether they mutate. Mutation is direct attribute assignment: `self.db.tasks[task_id] = task`, `task.status = status`. Errors are raised as `ValueError` and become error tool messages.
3. Tool schemas shown to the model are derived from the method signature and the Google-style docstring; there is no separate JSON schema file.
4. Non-tool helper methods on the same class (`assert_task_status`, `assert_number_of_tasks`, `add_notification`, `assert_notification_status`) are callable by name from task JSON via `env_type` + `func_name`, either as `initialization_actions` or `env_assertions`.
5. Initial state is loaded by `get_environment()` reading `db.json`, `user_db.json`, and `policy.md` from disk each time. Reset is "construct a new Environment". Per-task variation comes from `initial_state.initialization_data` (partial overlay merged into the DB), `initialization_actions` (helper calls), and `message_history` (a prefix conversation whose mutating tool calls are replayed via `set_state`).
6. A task is a JSON record: `user_scenario` (persona + instructions for the LLM user simulator), `ticket` (a solo-mode variant with no user), optional `initial_state`, and `evaluation_criteria`.
7. Verification runs after the episode. `EnvironmentEvaluator` builds two fresh environments: "predicted" replays the whole trajectory's mutating tool calls; "gold" replays only the task's `initial_state` and then executes the golden `actions`. The two DB hashes are compared for equality. Any agent path that reaches the same end state passes; the golden action list is not a script unless `ACTION` is in `reward_basis`.
8. `env_assertions` run on the predicted environment and call domain helper methods that return bools, compared against `assert_value`.
9. `ActionEvaluator` checks that every golden action has a matching tool call somewhere in the trajectory, using `compare_with_tool_call`: same name and equal arguments over `compare_args` (all args if `None`, and always true if `[]`, which is how `impossible_task_1` accepts any `summary` text).
10. `communicate_info` is substring matching over agent messages; `nl_assertions` is an LLM judge. Both are evaluated but only gate reward if listed in `reward_basis` (default `[DB, COMMUNICATE]`).
11. Final reward is the product of the gated components, so it is binary in practice.
12. Hard-coded: tool code, data model, helper assertions, hashing. Data-driven: the seed DB, policy text, every task, the split lists, and which reward components apply.
13. The simulated user is an LLM prompted with `user_scenario` and can itself call `user_tools`; its tool calls also count toward replay and action checks.

---

# 2. OpenEnv `echo_env` (huggingface/OpenEnv, formerly meta-pytorch/OpenEnv)

Tree of `envs/echo_env/`: `README.md`, `__init__.py`, `client.py`, `openenv.yaml`, `pyproject.toml`, `server/Dockerfile`, `server/__init__.py`, `server/app.py`, `server/echo_environment.py`, `uv.lock`.

## 2.1 `envs/echo_env/openenv.yaml`

```yaml
spec_version: 1
name: echo_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

## 2.2 `envs/echo_env/__init__.py`

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment - A pure MCP environment for testing and demonstration.

This environment exposes all functionality through MCP tools:
- `echo_message(message)`: Echo back the provided message
- `echo_with_length(message)`: Echo back the message with its length

Example:
    >>> from echo_env import EchoEnv
    >>>
    >>> with EchoEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...     tools = env.list_tools()
    ...     result = env.call_tool("echo_message", message="Hello!")
    ...     print(result)  # "Hello!"
"""

# Re-export MCP types for convenience
from openenv.core.env_server.mcp_types import CallToolAction, ListToolsAction

from .client import EchoEnv

__all__ = ["EchoEnv", "CallToolAction", "ListToolsAction"]
```

## 2.3 `envs/echo_env/server/echo_environment.py` (the environment)

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment Implementation.

A pure MCP environment that echoes back messages sent to it.
This demonstrates how to build an MCPEnvironment with inline FastMCP tools.

All interactions happen through MCP tools:
- `echo_message(message)`: Echo back the provided message
- `echo_with_length(message)`: Echo back the message with its length

Example:
    >>> from openenv.core.env_server.mcp_types import ListToolsAction, CallToolAction
    >>> env = EchoEnvironment()
    >>> env.reset()
    >>>
    >>> # List available tools
    >>> obs = env.step(ListToolsAction())
    >>> print([t.name for t in obs.tools])  # ["echo_message", "echo_with_length"]
    >>>
    >>> # Call a tool
    >>> obs = env.step(CallToolAction(tool_name="echo_message", arguments={"message": "Hi!"}))
    >>> print(obs.result)  # "Hi!"
"""

from typing import Any, Optional
from uuid import uuid4

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.mcp_environment import MCPEnvironment
    from openenv.core.env_server.types import Action, Observation, State

from fastmcp import FastMCP


class EchoEnvironment(MCPEnvironment):
    """
    A pure MCP echo environment that echoes back messages.

    This environment exposes all functionality through MCP tools:
    - `echo_message`: Echo back the provided message
    - `echo_with_length`: Echo back the message with its length

    The environment inherits MCP support (ListToolsAction, CallToolAction)
    from the MCPEnvironment base class. No legacy action types are supported.

    Example using MCPToolClient:
        >>> from openenv.core.mcp_client import MCPToolClient
        >>>
        >>> with MCPToolClient(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...     tools = env.list_tools()
        ...     print([t.name for t in tools])
        ...     result = env.call_tool("echo_message", message="Hello!")
        ...     print(result)
    """

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        """Initialize the echo environment with MCP server and tools."""
        # Create MCP server and define tools inline
        mcp = FastMCP("echo_env")

        @mcp.tool
        def echo_message(message: str) -> str:
            """
            Echo back the provided message.

            Args:
                message: The message to echo back

            Returns:
                The same message that was provided
            """
            return message

        @mcp.tool
        def echo_with_length(message: str) -> dict:
            """
            Echo back the message with its length.

            Args:
                message: The message to echo back

            Returns:
                Dictionary with the message and its length
            """
            return {"message": message, "length": len(message)}

        # Pass the MCP server to the base class
        super().__init__(mcp)
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Reset the environment.

        Args:
            seed: Optional random seed (unused in echo env)
            episode_id: Optional episode ID to use
            **kwargs: Additional reset options

        Returns:
            Observation indicating the environment is ready
        """
        self._state = State(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
        )
        self._reset_count += 1

        return Observation(
            done=False,
            reward=0.0,
            metadata={"status": "ready", "message": "Echo environment ready!"},
        )

    def _step_impl(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Handle non-MCP actions.

        This environment only supports MCP actions (ListToolsAction, CallToolAction).
        Any other action type returns an error observation.

        Args:
            action: The action to execute
            timeout_s: Optional timeout (unused)
            **kwargs: Additional arguments

        Returns:
            Observation with error for unknown action types
        """
        return Observation(
            done=False,
            reward=0.0,
            metadata={
                "error": f"Unknown action type: {type(action).__name__}. "
                "Use ListToolsAction or CallToolAction for MCP interactions."
            },
        )

    def step(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Execute a step in the environment.

        Delegates to base class for MCP actions. Increments step count for all actions.

        Args:
            action: The MCP action to execute (ListToolsAction or CallToolAction)
            timeout_s: Optional timeout for the action
            **kwargs: Additional arguments

        Returns:
            Observation from the action execution
        """
        # Increment step count for all actions
        self._state.step_count += 1

        # Let the base class handle MCP actions and non-MCP routing
        return super().step(action, timeout_s=timeout_s, **kwargs)

    async def step_async(
        self,
        action: Action,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> Observation:
        """
        Async step used by the WebSocket handler.

        Increments step count then delegates to MCPEnvironment.step_async,
        which routes MCP actions without going through run_async_safely.
        """
        self._state.step_count += 1
        return await super().step_async(action, timeout_s=timeout_s, **kwargs)

    @property
    def state(self) -> State:
        """
        Get the current environment state.

        Returns:
            Current State with episode_id and step_count
        """
        return self._state
```

## 2.4 `envs/echo_env/server/app.py` (HTTP/WebSocket server)

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application for the Echo Environment.

This module creates an HTTP server that exposes the EchoEnvironment
over HTTP and WebSocket endpoints, compatible with MCPToolClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    uv run --project . server
"""

import os

# Support both in-repo and standalone imports
try:
    # In-repo imports (when running from OpenEnv repository)
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

    from .echo_environment import EchoEnvironment
except ImportError:
    # Standalone imports (when environment is standalone with openenv from pip)
    from openenv.core.env_server.http_server import create_app
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation
    from server.echo_environment import EchoEnvironment

# Create the app with web interface and README integration
# Pass the class (factory) instead of an instance for WebSocket session support
# Use MCP types for action/observation since this is a pure MCP environment
max_concurrent = int(os.getenv("MAX_CONCURRENT_ENVS", "8"))

app = create_app(
    EchoEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="echo_env",
    max_concurrent_envs=max_concurrent,
)


def main():
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        python -m envs.echo_env.server.app
        openenv serve echo_env

    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
```

## 2.5 `envs/echo_env/server/__init__.py`

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Echo environment server components."""

from .echo_environment import EchoEnvironment

__all__ = ["EchoEnvironment"]
```

## 2.6 `envs/echo_env/client.py`

```python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Echo Environment Client.

This module provides the client for connecting to an Echo Environment server.
EchoEnv extends MCPToolClient to provide tool-calling style interactions.

Example:
    >>> with EchoEnv(base_url="http://localhost:8000") as env:
    ...     env.reset()
    ...
    ...     # Discover tools
    ...     tools = env.list_tools()
    ...     print([t.name for t in tools])  # ['echo_message', 'echo_with_length']
    ...
    ...     # Call tools
    ...     result = env.call_tool("echo_message", message="Hello!")
    ...     print(result)  # "Hello!"
    ...
    ...     result = env.call_tool("echo_with_length", message="Test")
    ...     print(result)  # {"message": "Test", "length": 4}
"""

from openenv.core.mcp_client import MCPToolClient


class EchoEnv(MCPToolClient):
    """
    Client for the Echo Environment.

    This client provides a simple interface for interacting with the Echo
    Environment via MCP tools. It inherits all functionality from MCPToolClient:
    - `list_tools()`: Discover available tools
    - `call_tool(name, **kwargs)`: Call a tool by name
    - `reset(**kwargs)`: Reset the environment
    - `step(action)`: Execute an action (for advanced use)

    Example:
        >>> # Connect to a running server
        >>> with EchoEnv(base_url="http://localhost:8000") as env:
        ...     env.reset()
        ...
        ...     # List available tools
        ...     tools = env.list_tools()
        ...     for tool in tools:
        ...         print(f"{tool.name}: {tool.description}")
        ...
        ...     # Echo a message
        ...     result = env.call_tool("echo_message", message="Hello!")
        ...     print(result)  # "Hello!"
        ...
        ...     # Echo with length
        ...     result = env.call_tool("echo_with_length", message="Test")
        ...     print(result)  # {"message": "Test", "length": 4}

    Example with Docker:
        >>> # Automatically start container and connect
        >>> env = EchoEnv.from_docker_image("echo-env:latest")
        >>> try:
        ...     env.reset()
        ...     tools = env.list_tools()
        ...     result = env.call_tool("echo_message", message="Hello!")
        ... finally:
        ...     env.close()

    Example with HuggingFace Space:
        >>> # Run from HuggingFace Space
        >>> env = EchoEnv.from_env("openenv/echo-env")
        >>> try:
        ...     env.reset()
        ...     result = env.call_tool("echo_message", message="Hello!")
        ... finally:
        ...     env.close()
    """

    pass  # MCPToolClient provides all needed functionality
```

## 2.7 Action / Observation / State models

echo_env defines none of its own; it uses the framework's. From `src/openenv/core/env_server/types.py` (three base classes excerpted from a 400+ line file):

```python
class Action(BaseModel):
    """Base class for all environment actions.

    All action subclasses should inherit from this base class.
    Uses Pydantic for automatic validation and serialization.
    """

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown fields
        validate_assignment=True,  # Validate on field assignment
        arbitrary_types_allowed=True,  # Allow numpy arrays, torch tensors, etc.
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the action"
    )


class Observation(BaseModel):
    """Base class for all environment observations.

    All observation subclasses should inherit from this base class.
    Uses Pydantic for automatic validation and serialization.
    """

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    done: bool = Field(default=False, description="Whether the episode has terminated")
    reward: bool | int | float | None = Field(
        default=None, description="Reward signal from the last action"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata for the observation"
    )


class State(BaseModel):
    """Base class for environment state.

    Represents internal environment state, separate from observations.
    """

    model_config = ConfigDict(
        extra="allow",  # Allow extra fields for flexibility
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )

    episode_id: Optional[str] = Field(
        default=None, description="Unique identifier for the current episode"
    )
    step_count: int = Field(
        default=0,
        ge=0,  # Greater than or equal to 0
        description="Number of steps taken in the current episode",
    )
```

From `src/openenv/core/env_server/mcp_types.py` (the tool, action, and observation classes; the JSON-RPC envelope classes and WebSocket message types in the same file are omitted):

```python
class Tool(BaseModel):
    """
    Strongly typed MCP tool specification.

    Follows the MCP ToolSpec format for tool discovery.
    See: https://modelcontextprotocol.io/specification/2025-06-18/server/tools
    """

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Unique identifier for the tool")
    description: str = Field(
        description="Human-readable description of what the tool does"
    )
    input_schema: Dict[str, Any] = Field(
        description="JSON Schema for the tool's input parameters"
    )


class ToolErrorType(str, Enum):
    """Types of errors that can occur during tool execution."""

    EXECUTION_ERROR = "execution_error"  # Tool ran but failed
    INVALID_ARGS = "invalid_args"  # Invalid arguments provided
    TRANSPORT_ERROR = "transport_error"  # Communication failure
    TOOL_NOT_FOUND = "tool_not_found"  # Tool doesn't exist
    TIMEOUT = "timeout"  # Operation timed out


class ToolError(BaseModel):
    """
    Structured error for tool execution failures.

    This is used for transport/framework errors, NOT for errors returned
    by the tool itself (those go in the result field).
    """

    model_config = ConfigDict(extra="forbid")

    error_type: ToolErrorType = Field(description="Category of the error")
    message: str = Field(description="Human-readable error message")


# --- MCP Actions ---


class ListToolsAction(Action):
    """
    Request list of available tools from the environment.

    This action triggers MCP's tools/list operation and returns
    all available tools with their schemas. Does NOT require reset()
    to be called first.
    """

    type: Literal["list_tools"] = Field(
        default="list_tools", description="Action type discriminator"
    )


class CallToolAction(Action):
    """
    Call a specific tool via MCP.

    This action triggers MCP's tools/call operation with the
    specified tool name and arguments.
    """

    type: Literal["call_tool"] = Field(
        default="call_tool", description="Action type discriminator"
    )
    tool_name: str = Field(description="Name of the tool to call")
    arguments: Dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool"
    )


# --- MCP Observations ---


class ListToolsObservation(Observation):
    """
    Response containing available tools.

    Returned when processing a ListToolsAction.
    """

    tools: List[Tool] = Field(description="List of available tools with their schemas")


class CallToolObservation(Observation):
    """
    Response from tool execution.

    Contains the tool's result or an error if the call failed.
    Tool-specific errors (from the tool itself) are included in the result.
    Transport/framework errors use the error field.
    """

    tool_name: str = Field(description="Name of the tool that was called")
    result: Any = Field(
        default=None, description="Tool-specific result (may include tool errors)"
    )
    error: Optional[ToolError] = Field(
        default=None, description="Transport/framework error if call failed"
    )
```

Also in that file: `RESERVED_TOOL_NAMES = frozenset(["reset", "step", "state", "close"])`, which prevents a tool from shadowing the env API.

## 2.8 `envs/echo_env/README.md`

```markdown
---
title: Echo Environment Server
emoji: 🔊
colorFrom: blue
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Echo Environment

A simple test environment that echoes back messages. Perfect for testing the env APIs as well as demonstrating environment usage patterns.

## Quick Start

The simplest way to use the Echo environment is through the `EchoEnv` class. The client is **async by default**:

```python
import asyncio
from echo_env import CallToolAction, EchoEnv

async def main():
    # Create environment from Docker image
    client = await EchoEnv.from_docker_image("echo-env:latest")

    async with client:
        await client.reset()

        # Send multiple messages
        messages = ["Hello, World!", "Testing echo", "Final message"]

        for msg in messages:
            result = await client.step(
                CallToolAction(
                    tool_name="echo_message",
                    arguments={"message": msg},
                )
            )
            print(f"Sent: '{msg}'")
            print(f"  → Echoed: '{result.observation.result}'")
            print(f"  → Reward: {result.reward}")

asyncio.run(main())
```

For **synchronous usage**, use the `.sync()` wrapper:

```python
from echo_env import CallToolAction, EchoEnv

with EchoEnv(base_url="http://localhost:8000").sync() as client:
    client.reset()
    result = client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello!"},
        )
    )
    print(result.observation.result)
```

The `EchoEnv.from_docker_image()` method handles:
- Starting the Docker container
- Waiting for the server to be ready
- Connecting to the environment
- Container cleanup when the context manager exits

## Building the Docker Image

Before using the environment, you need to build the Docker image:

```bash
# From project root
docker build -t echo-env:latest -f envs/echo_env/server/Dockerfile .
```

## Environment Details

### Tools
- `echo_message(message)` - Echo the provided message
- `echo_with_length(message)` - Echo the message and include its length

### Observation
**CallToolObservation**: Contains the tool result and metadata
- `result` - The tool return value
- `reward` (float) - Reward returned by the environment
- `done` (bool) - Always `False` for the echo environment
- `metadata` (dict) - Additional info like step count

### Reward
The echo environment returns `0.0` reward for tool calls. It is intended as a
minimal MCP integration example rather than a reward-shaping reference.

## Advanced Usage

### Connecting to an Existing Server

If you already have an Echo environment server running, you can connect directly:

```python
from echo_env import CallToolAction, EchoEnv

# Async usage
async with EchoEnv(base_url="http://localhost:8000") as client:
    await client.reset()
    result = await client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello!"},
        )
    )

# Sync usage
with EchoEnv(base_url="http://localhost:8000").sync() as client:
    client.reset()
    result = client.step(
        CallToolAction(
            tool_name="echo_message",
            arguments={"message": "Hello!"},
        )
    )
```

Note: When connecting to an existing server, closing the client will NOT stop the server.

## Development & Testing

### Direct Environment Testing

Test the environment logic directly without starting the HTTP server:

```bash
# From the server directory
python3 envs/echo_env/server/test_echo_env.py
```

This verifies that:
- Environment resets correctly
- Step executes actions properly
- State tracking works
- Rewards are calculated correctly

### Running the Full Example

Run the complete example that demonstrates the full workflow:

```bash
python3 examples/local_echo_env.py
```

This example shows:
- Creating an environment from a Docker image
- Resetting and stepping through the environment
- Automatic cleanup with `close()`

## Project Structure

```
echo_env/
├── __init__.py            # Module exports
├── README.md              # This file
├── client.py              # EchoEnv client implementation
├── models.py              # Action and Observation models
└── server/
    ├── __init__.py        # Server module exports
    ├── echo_environment.py  # Core environment logic
    ├── app.py             # FastAPI application
    ├── test_echo_env.py   # Direct environment tests
    └── Dockerfile         # Container image definition
```
```

(The README's tree lists `models.py` and `test_echo_env.py`; neither exists in the current `main` tree.)

## 2.9 Commentary: OpenEnv echo_env

1. State is a single Pydantic `State(episode_id, step_count)` held in `self._state`. The env has no domain state at all; it exists to show the transport.
2. Tools are `@mcp.tool` closures on a `FastMCP` server created inside `__init__`. The docstring and type hints become the MCP `input_schema`. Tools do not touch env state here; in a real env they would close over `self`.
3. `MCPEnvironment` (base class, not shown, 24 KB) turns `ListToolsAction` into `tools/list` and `CallToolAction` into `tools/call` against that in-process FastMCP server, and wraps results in `CallToolObservation(tool_name, result, error, reward, done, metadata)`.
4. The Gym-style surface is `reset(seed, episode_id)`, `step(action)`, and the `state` property. Both `step` and `step_async` exist because the server can be driven over HTTP or a WebSocket session.
5. Reset is trivial: mint a new `episode_id`, zero `step_count`, return an `Observation` with `reward=0.0, done=False` and a status message. Nothing is loaded from disk.
6. There is no task and no verifier. `reward` is always 0.0 and `done` is always False. Success is simply "the tool call returned". This is the smallest example of the OpenEnv contract, not of grading.
7. Serving: `create_app(EchoEnvironment, CallToolAction, CallToolObservation, ...)` takes the class (a factory) so each WebSocket session gets its own instance; `MAX_CONCURRENT_ENVS` caps them. `openenv.yaml` is the Space/CLI manifest.
8. The client is an empty subclass of `MCPToolClient`, which provides `from_docker_image`, `from_env` (HF Space), `list_tools`, `call_tool`, `reset`, `step`, and a `.sync()` wrapper.
9. Hard-coded: everything (tool bodies, reward, done). Data-driven: nothing beyond env vars. Other OpenEnv envs (e.g. `calendar_env`, `finqa_env`) add task lists via `ListTasksRequest`/`GetTaskRequest` types that exist in `types.py`, but echo_env does not use them.
10. No user simulator; the "user" is whatever RL loop calls `step`.

---

# 3. Prime Intellect `verifiers`, `environments/wiki_search`

Tree: `environments/wiki_search/pyproject.toml`, `wiki_search/__init__.py`, `wiki_search/taskset.py`, `wiki_search/judge.py`, `wiki_search/judge_prompt.txt`, `wiki_search/servers/__init__.py` (empty), `wiki_search/servers/wiki.py`. This is the current `verifiers.v1` API (Taskset / Toolset / Judge), which differs from the older `ToolEnv` API.

## 3.1 `wiki_search/__init__.py`

```python
from wiki_search.judge import WikiSearchJudge
from wiki_search.taskset import WikiSearchTaskset

__all__ = ["WikiSearchJudge", "WikiSearchTaskset"]
```

## 3.2 `wiki_search/taskset.py` (task definition and wiring)

```python
"""Answer trivia with a worker-shared wiki corpus and a reference judge.

The tool server builds an expensive corpus/index in its process-level `setup`, so
`Taskset.toolsets` launches one instance per environment worker instead of rebuilding it
per rollout. The tools are read-only; grading comes from the plugged wiki-search judge
(class-level prompt; reject incoherent answers).
"""

from pydantic import Field

import verifiers.v1 as vf
from wiki_search.judge import WikiSearchJudgeConfig
from wiki_search.servers.wiki import WikiSearchToolset

SYSTEM = (
    "Use the Wikipedia search tools — `wiki_search_pages` to find relevant pages, "
    "`wiki_view_sections` to list a page's sections, and `wiki_read_section` to read "
    "one — to answer the question. When confident, reply with a concise final answer."
)

# The question bank and count are fixed properties of this env, not eval-time
# knobs (the searchable corpus is built in `WikiSearchToolset.setup`).
QUESTIONS_DATASET = "willcb/wiki-trivia-questions-v4"
NUM_QUESTIONS = 20


class WikiSearchTaskConfig(vf.TaskConfig):
    # Users can replace or reconfigure this judge through --taskset.task.judges.
    judges: vf.Judges = Field(default_factory=lambda: [WikiSearchJudgeConfig()])


class TriviaTaskData(vf.TaskData):
    # These fields feed the reference judge's question and answer template values.
    question: str
    answer: str


class TriviaTask(vf.Task[TriviaTaskData, vf.State, WikiSearchTaskConfig]):
    pass


class WikiSearchConfig(vf.TasksetConfig):
    tools: vf.SharedToolsetConfig = vf.SharedToolsetConfig()
    task: WikiSearchTaskConfig = WikiSearchTaskConfig()


class WikiSearchTaskset(vf.Taskset[TriviaTask, WikiSearchConfig]):
    @classmethod
    def toolsets(cls, config: WikiSearchConfig) -> list[vf.Toolset]:
        return [WikiSearchToolset(config.tools)]

    def load(self) -> list[TriviaTask]:
        from datasets import load_dataset

        rows = load_dataset(QUESTIONS_DATASET, split="train")
        return [
            TriviaTask(
                TriviaTaskData(
                    idx=i,
                    question=row["question"],
                    answer=str(row["answer"]),
                    prompt=f"{SYSTEM}\n\nQuestion: {row['question']}",
                ),
                self.config.task,
            )
            for i, row in enumerate(rows.select(range(min(NUM_QUESTIONS, len(rows)))))
        ]
```

## 3.3 `wiki_search/servers/wiki.py` (the tool server)

```python
import verifiers.v1 as vf

DATASET = "willcb/rare-wiki-pages"


class WikiSearchToolset(vf.Toolset[vf.SharedToolsetConfig]):
    """Read-only search/view/read over the wiki corpus. The corpus + chroma index (expensive) are
    built once in `setup`, in the server process. Every tool call is a read."""

    TOOL_PREFIX = "wiki"  # the model sees `wiki_search_pages` / `wiki_view_sections` / `wiki_read_section`

    async def setup(self) -> None:
        import os
        from pathlib import Path

        # chromadb (via opentelemetry) imports protobuf-generated code predating protobuf 6.x;
        # force the pure-Python impl so it imports next to a newer protobuf (e.g. vllm's).
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        import chromadb
        from datasets import load_dataset

        rows = load_dataset(DATASET, split="train")
        self.pages = {
            r["id"]: {"title": r["title"], "content": r["content"]} for r in rows
        }
        cache = os.environ.get(
            "WIKI_SEARCH_CACHE", str(Path.home() / ".cache" / "wiki_search")
        )
        col = chromadb.PersistentClient(
            path=f"{cache}/chroma"
        ).get_or_create_collection(
            "wiki_titles"  # default local embedder (onnx MiniLM) — no API key
        )
        ids = list(self.pages)
        have: set[str] = set()
        for i in range(0, len(ids), 500):
            have.update(col.get(ids=ids[i : i + 500]).get("ids", []))
        missing = [pid for pid in ids if pid not in have]
        for i in range(0, len(missing), 256):
            batch = missing[i : i + 256]
            col.upsert(ids=batch, documents=[self.pages[pid]["title"] for pid in batch])
        self.collection = col

    @staticmethod
    def _norm(text: str) -> str:
        """lowercase, spaces -> underscores (section ids)."""
        return text.strip().lower().replace(" ", "_")

    @vf.tool
    def search_pages(self, query: str) -> list[dict]:
        """Search the wiki for the 10 most relevant pages (by title). Returns
        `[{page_id, title}, ...]` — pass a page_id to view_sections."""
        result = self.collection.query(query_texts=[query], n_results=10)
        return [
            {"page_id": pid, "title": self.pages[pid]["title"]}
            for pid in result["ids"][0]
        ]

    @vf.tool
    def view_sections(self, page_id: str) -> list[dict]:
        """List a page's sections. Returns `[{section_id, section_name}, ...]` — pass a
        section_id to read_section."""
        content = self.pages[page_id]["content"]
        sections = [
            {
                "section_id": f"{page_id}:{self._norm(line.lstrip('#').strip())}",
                "section_name": line.lstrip("#").strip(),
            }
            for line in content.split("\n")
            if line.startswith("#")
        ]
        return sections or [
            {"section_id": f"{page_id}:full", "section_name": "Full Page"}
        ]

    @vf.tool
    def read_section(self, section_id: str) -> str:
        """Read the text of a section (`page_id:section`; use `page_id:full` for all)."""
        page_id, section = section_id.split(":", 1)
        content = self.pages[page_id]["content"]
        if section == "full":
            return content
        lines = content.split("\n")
        start = end = None
        for i, line in enumerate(lines):
            if line.startswith("#"):
                if self._norm(line.lstrip("#").strip()) == section and start is None:
                    start = i
                elif start is not None:
                    end = i
                    break
        if start is None:
            return f"Section not found: {section_id}"
        return "\n".join(lines[start : end or len(lines)])


if __name__ == "__main__":
    WikiSearchToolset.run()
```

## 3.4 `wiki_search/judge.py` (the verifier)

```python
"""Wiki-search reference judge: same scoring as `reference`, env-specific prompt."""

from pathlib import Path

import verifiers.v1 as vf

JUDGE_PROMPT = (Path(__file__).parent / "judge_prompt.txt").read_text()


class WikiSearchJudgeConfig(vf.ReferenceJudgeConfig):
    id: vf.ID = "wiki-search"
    """Local package id — `load_judge` resolves this module's `WikiSearchJudge`."""
    question_field: str = "question"


class WikiSearchJudge(vf.ReferenceJudge):
    prompt = JUDGE_PROMPT


__all__ = ["WikiSearchJudge", "WikiSearchJudgeConfig"]
```

## 3.5 `wiki_search/judge_prompt.txt`

```text
Given a ground truth answer and a response, determine if the response is both correct and coherent.

Question:
```
{question}
```

Ground truth answer:
```
{answer}
```

Response:
```
{response}
```

Respond either "yes" or "no" only. If a response contains incoherent text, respond with "no" even if the correct answer is also present.
```

## 3.6 Commentary: verifiers wiki_search

1. There is no mutable environment state. The tool server's state (`self.pages` dict and a Chroma collection) is built once per worker in `async setup()` and is read-only afterwards. The per-rollout state is the framework's `vf.State` (the chat transcript), which the env does not subclass.
2. Tools are `@vf.tool` methods on a `vf.Toolset` subclass. `TOOL_PREFIX = "wiki"` namespaces them, so the model sees `wiki_search_pages`, etc. The docstring is the tool description. `WikiSearchToolset.run()` under `__main__` means the toolset is a standalone process the framework launches and talks to over its own RPC; `SharedToolsetConfig` means one instance is shared by all rollouts on a worker rather than spawned per episode.
3. Tools never mutate anything; the search is over page titles via a local ONNX MiniLM embedder in Chroma, and `read_section` slices markdown by heading.
4. Initial state per task is just a prompt: `SYSTEM + "\n\nQuestion: " + question`. Reset is implicit: a new rollout starts from that prompt with the same shared tool server.
5. A task is a `TriviaTask` built from a Hugging Face dataset row (`willcb/wiki-trivia-questions-v4`), holding `question`, `answer`, and `prompt`. `NUM_QUESTIONS = 20` caps the set. Task data fields are what the judge template can reference.
6. Verification is an LLM judge. `WikiSearchJudge` subclasses `vf.ReferenceJudge`, which formats `judge_prompt.txt` with `{question}`, `{answer}`, `{response}` and scores "yes"/"no". The config field `question_field = "question"` tells the base judge which task field to inject. Judges are swappable from the CLI via `--taskset.task.judges`.
7. No trajectory or tool-call checking exists here; only the final response is graded. Whether the agent actually used the tools is not verified.
8. Hard-coded: the tool implementations, the dataset names, the question cap, the system prompt, and the judge prompt text. Data-driven: the questions and reference answers (from HF), and the judge selection via config.
9. No user simulator. The `nemo_gym_weather` sibling env is even smaller (a 20-line `taskset.py` plus a 5-row JSONL with `verifier_metadata: {"expected_city": ...}`), but it delegates both tools and scoring to NeMo Gym's resource server, so it shows less.

---

# Comparison

| Dimension | tau2-bench mock | OpenEnv echo_env | verifiers wiki_search |
|---|---|---|---|
| State store | In-memory Pydantic `MockDB` + `MockUserDB` loaded from `db.json` / `user_db.json`; hashed for comparison | `State(episode_id, step_count)` only; no domain state | Read-only corpus dict + Chroma index in a shared tool-server process; rollout state is the transcript |
| Tool style | Methods on `ToolKitBase` with `@is_tool(ToolType.READ/WRITE/GENERIC)`; schema from signature + docstring; mutate `self.db` directly | `@mcp.tool` closures on an in-process FastMCP server; invoked via `CallToolAction` → `CallToolObservation` | `@vf.tool` methods on a `vf.Toolset` run as a separate process; prefixed names; all reads |
| Task spec | JSON record: `user_scenario` (persona + instructions), `ticket`, `initial_state` (data overlay / init actions / message prefix), `evaluation_criteria` | None in echo_env (framework has `ListTasksRequest`/`GetTaskRequest` types for envs that want them) | `TriviaTask(question, answer, prompt)` built from an HF dataset row in `Taskset.load()` |
| Verifier | Replay-and-compare: gold env runs golden `actions`, predicted env replays trajectory, DB hashes compared; plus `env_assertions` (bool helper methods), `actions` match via `compare_with_tool_call`, `communicate_info` substring, `nl_assertions` LLM judge; product of components in `reward_basis` | None; `reward=0.0`, `done=False` always | LLM `ReferenceJudge` with a yes/no prompt over `{question, answer, response}`; final answer only |
| Reset mechanism | Construct a fresh `Environment` from disk, then `set_state(initialization_data, initialization_actions, message_history)` replays mutating tool calls with output checks | `reset()` mints a new `episode_id`, zeroes `step_count`; nothing loaded | Implicit: new rollout starts from the task prompt; tool server persists across rollouts (`setup()` runs once per worker) |
| User simulator | Yes: LLM user driven by `user_scenario`, may call `user_tools`; solo mode replaces it with `ticket` | No | No |
| What is hard-coded vs data | Code: tools, models, assertion helpers. Data: seed DB, policy, tasks, splits, reward basis | Everything is code; env vars only | Code: tools, prompts, dataset ids, cap. Data: questions/answers; judge choice via config |
| Reward shape | Binary per task (product of 0/1 components) | Constant 0.0 | Binary per task (judge yes/no), swappable |

Sources fetched (all raw from `main`): `github.com/sierra-research/tau2-bench` under `src/tau2/domains/mock/`, `data/tau2/domains/mock/`, `src/tau2/evaluator/`, `src/tau2/environment/`, `src/tau2/data_model/tasks.py`; `github.com/huggingface/OpenEnv` under `envs/echo_env/` and `src/openenv/core/env_server/`; `github.com/PrimeIntellect-ai/verifiers` under `environments/wiki_search/` and `environments/nemo_gym_weather/`.