from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import logging
import os
import re
import shlex
import time
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal

import yaml
from jinja2 import Template
from pydantic import BaseModel, ConfigDict, Field, model_validator
from simple_parsing.helpers.fields import field
from swerex.exceptions import BashIncorrectSyntaxError, CommandTimeoutError, SwerexException
from tenacity import RetryError
from typing_extensions import Self
from unidiff import UnidiffParseError

from sweagent import __version__, get_agent_commit_hash, get_rex_commit_hash, get_rex_version
from sweagent.agent.action_sampler import AbstractActionSampler, ActionSamplerConfig
from sweagent.agent.history_processors import DefaultHistoryProcessor, HistoryProcessor
from sweagent.agent.hooks.abstract import AbstractAgentHook, CombinedAgentHook
from sweagent.agent.models import (
    AbstractModel,
    HumanModel,
    HumanThoughtModel,
    InstanceStats,
    ModelConfig,
    get_model,
)
from sweagent.agent.problem_statement import ProblemStatement, ProblemStatementConfig
from sweagent.agent.reviewer import (
    ChooserRetryLoop,
    RetryLoopConfig,
    ReviewSubmission,
    ScoreRetryLoop,
    get_retry_loop_from_config,
)
from sweagent.environment.repo import runtime_repo_path
from sweagent.environment.swe_env import SWEEnv
from sweagent.exceptions import (
    BudgetExhaustedError,
    ContentPolicyViolationError,
    ContextWindowExceededError,
    CostLimitExceededError,
    FormatError,
    TotalCostLimitExceededError,
)
from sweagent.tools.parsing import (
    ActionOnlyParser,
    ThoughtActionParser,
)
from sweagent.tools.tools import ToolConfig, ToolHandler
from sweagent.types import AgentInfo, AgentRunResult, StepOutput, Trajectory, TrajectoryStep
from sweagent.utils.config import _convert_paths_to_abspath, _strip_abspath_from_dict
from sweagent.utils.jinja_warnings import _warn_probably_wrong_jinja_syntax
from sweagent.utils.log import get_logger
from sweagent.utils.patch_formatter import PatchFormatter


def _model_patch_path() -> str:
    return os.getenv("SWE_AGENT_MODEL_PATCH_PATH", "/tmp/model.patch")


class TemplateConfig(BaseModel):
    """This configuration is used to define almost all message templates that are
    formatted by the agent and sent to the LM.
    """

    system_template: str = ""
    instance_template: str = ""
    next_step_template: str = "Observation: {{observation}}"

    next_step_truncated_observation_template: str = (
        "Observation: {{observation[:max_observation_length]}}<response clipped>"
        "<NOTE>Observations should not exceeded {{max_observation_length}} characters. "
        "{{elided_chars}} characters were elided. Please try a different command that produces less output "
        "or use head/tail/grep/redirect the output to a file. Do not use interactive pagers.</NOTE>"
    )
    """Message template for when the agent's observation was truncated.
    Available variables: `observation`, `max_observation_length`, `elided_chars`
    """

    max_observation_length: int = 100_000
    """Truncate observation to this length if it exceeds it.
    This in measured in characters, i.e., as `len(observation)`.
    """

    next_step_no_output_template: str = None  # type: ignore
    """Template for the next step when the last output was empty. Defaults to next_step_template."""

    strategy_template: str | None = None
    demonstration_template: str | None = None

    demonstrations: list[Path] = field(default_factory=list)
    """Paths to demonstrations. If path is not absolute, it is assumed to be
    relative to the SWE_AGENT_CONFIG_ROOT (if set) or the SWE-agent repository root
    """

    put_demos_in_history: bool = False
    """If True, add demonstration to history instead of as a single message"""

    disable_image_processing: bool = False
    """If True, disable image processing for multimodal problem statements (i.e. SWEBenchMultimodalProblemStatement).
    """

    shell_check_error_template: str = (
        "Your bash command contained syntax errors and was NOT executed. "
        "Please fix the syntax errors and try again. This can be the result "
        "of not adhering to the syntax for multi-line commands. Here is the output of `bash -n`:\n"
        "{{bash_stdout}}\n{{bash_stderr}}"
    )
    """Message template for when the agent's bash command contains syntax errors.
    Available variables: `bash_stdout`, `bash_stderr`
    """

    command_cancelled_timeout_template: str = (
        "The command '{{command}}' was cancelled because it took more than {{timeout}} seconds. "
        "Please try a different command that completes more quickly. "
        "Note: A common source of this error is if the command is interactive or requires user input "
        "(it is impossible to receive user input in the current environment, so the command will never complete)."
    )
    """Message template for when the agent's command was cancelled because it took too long.
    Available variables: `timeout`, `command`
    """

    def model_post_init(self, __context):
        self.demonstrations = _convert_paths_to_abspath(self.demonstrations)
        if self.next_step_no_output_template is None:
            self.next_step_no_output_template = self.next_step_template

    @model_validator(mode="after")
    def validate_template_jinja_syntax(self) -> Self:
        template_fields = [field for field in self.model_fields.keys() if field.endswith("_template")]
        for field in template_fields:
            value = getattr(self, field)
            _warn_probably_wrong_jinja_syntax(value)
        return self

    @model_validator(mode="after")
    def warnings(self) -> Self:
        logger = get_logger("swea-config", emoji="🔧")
        if self.put_demos_in_history and self.demonstration_template is not None:
            logger.warning("demonstration_template is ignored when put_demos_in_history is True")
        if not self.system_template or not self.instance_template:
            logger.warning(
                "system_template/instance_template is not set, using empty string. Perhaps you were"
                " overwriting the default config? See https://swe-agent.com/latest/usage/cl_tutorial/"
                " for more information. Note: You can ignore this warning in human mode."
            )
        return self


class DefaultAgentConfig(BaseModel):
    """This configuration object specifies the behavior of an agent."""

    name: str = "main"
    templates: TemplateConfig = Field(default_factory=TemplateConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    history_processors: list[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig = Field(description="Model options.")

    max_requeries: int = 3
    """Maximum number of times to requery the model after an error, such as a
    formatting error, a blocked action, or a bash syntax error.
    """
    action_sampler: ActionSamplerConfig | None = None

    type: Literal["default"] = "default"

    # pydantic config
    model_config = ConfigDict(extra="forbid")


class ShellAgentConfig(BaseModel):
    name: str = "main"
    templates: TemplateConfig = Field(default_factory=TemplateConfig)
    tools: ToolConfig = Field(default_factory=ToolConfig)
    history_processors: list[HistoryProcessor] = Field(default_factory=lambda: [DefaultHistoryProcessor()])
    model: ModelConfig = Field(description="Model options.")

    max_requeries: int = 3
    """Maximum number of times to requery the model after an error, such as a
    formatting error, a blocked action, or a bash syntax error.
    """

    type: Literal["shell"] = "shell"

    # pydantic config
    model_config = ConfigDict(extra="forbid")


class RetryAgentConfig(BaseModel):
    name: str = "retry_main"
    agent_configs: list[DefaultAgentConfig]
    retry_loop: RetryLoopConfig
    type: Literal["retry"] = "retry"
    model_config = ConfigDict(extra="forbid")


AgentConfig = Annotated[DefaultAgentConfig | RetryAgentConfig | ShellAgentConfig, Field(union_mode="left_to_right")]


class _BlockedActionError(Exception):
    """Raised when the agent's action is blocked"""


class _RetryWithOutput(Exception):
    """Used for internal control flow"""


class _RetryWithoutOutput(Exception):
    """Used for internal control flow"""


class _ExitForfeit(Exception):
    """Used for internal control flow"""


class _TotalExecutionTimeExceeded(Exception):
    """Used for internal control flow"""


RETRY_WITH_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITH-OUTPUT###"
RETRY_WITHOUT_OUTPUT_TOKEN = "###SWE-AGENT-RETRY-WITHOUT-OUTPUT###"
EXIT_FORFEIT_TOKEN = "###SWE-AGENT-EXIT-FORFEIT###"

THERMO_FRICTION_PATTERNS = (
    r"command not found",
    r"no such file or directory",
    r"syntax error",
    r"invalid option",
    r"not a git repository",
    r"permission denied",
    r"cancelled because it took more than",
)
THERMO_USEFUL_PATTERNS = (
    r"\b\d+\s+passed\b",
    r"all tests passed",
    r"==+\s*\d+ passed",
    r"\bok\b",
)

# ── Probe: action_taxonomy ──────────────────────────────────────────────
# Fine-grained classification of what the agent is doing each step.
_ACTION_TAXONOMY: list[tuple[str, re.Pattern[str]]] = [
    ("search",      re.compile(r"^(find|grep|rg|ag|search|find_file|locate)\b")),
    ("read_file",   re.compile(r"^(cat|head|tail|less|more|view)\b")),
    ("read_file",   re.compile(r"^str_replace_editor\s+view\b")),
    ("edit",        re.compile(r"^(sed|patch|edit)\b")),
    ("edit",        re.compile(r"^str_replace_editor\s+(str_replace|create|insert)\b")),
    ("navigate",    re.compile(r"^(cd|ls|pwd|tree)\b")),
    ("test",        re.compile(r"^(pytest|python\s+-m\s+pytest|tox)\b")),
    ("python",      re.compile(r"^python[23]?\b")),
    ("git",         re.compile(r"^git\b")),
    ("cleanup",     re.compile(r"^(rm|mv|cp)\b")),
    ("submit",      re.compile(r"^submit\b")),
    ("exit",        re.compile(r"^exit\b")),
]

# ── Probe: observation_bloat ────────────────────────────────────────────
# Thresholds for detecting oversized observations that waste context window.
_OBS_MEDIUM_THRESHOLD = 1000    # chars – above this is "medium"
_OBS_BLOAT_THRESHOLD = 4000     # chars – above this is "large"
_OBS_HUGE_THRESHOLD  = 10000    # chars – above this is "huge"

# ── Probe: iteration_waste ──────────────────────────────────────────────
# Patterns that suggest the agent is repeating itself or undoing work.
_ITERATION_WASTE_PATTERNS = (
    re.compile(r"^str_replace_editor\s+str_replace\b"),  # re-editing same file repeatedly
    re.compile(r"^git\s+(checkout|reset|stash|restore)\b"),       # reverting changes
)
_GIT_CHECK_PATTERN = re.compile(r"^git\s+(diff|status|log|show)\b")
_TEST_COMMAND_PATTERN = re.compile(r"^(pytest|python\s+-m\s+pytest|tox)\b")
_PYTHON_RUN_PATTERN = re.compile(r"^python[23]?\s")
_FILE_READ_PATTERN = re.compile(r"(?:cat|head|tail|less|more)\s+(\S+)|str_replace_editor\s+view\s+(\S+)")

# ── Probe: unknown_diagnostics ─────────────────────────────────────────
# Fine-grained diagnosis for thermo_probe="unknown" steps.
_UNKNOWN_ERROR_SIGNAL_PATTERNS = (
    re.compile(r"\btraceback\b", re.IGNORECASE),
    re.compile(r"\bexception\b", re.IGNORECASE),
    re.compile(r"\berror:\b", re.IGNORECASE),
    re.compile(r"\bassertionerror\b", re.IGNORECASE),
    re.compile(r"\b\d+\s+failed\b", re.IGNORECASE),
    re.compile(r"\bfailures\b", re.IGNORECASE),
)
_UNKNOWN_TEST_SIGNAL_PATTERNS = (
    re.compile(r"\bpytest\b", re.IGNORECASE),
    re.compile(r"\btox\b", re.IGNORECASE),
    re.compile(r"short test summary", re.IGNORECASE),
    re.compile(r"\bcollected\s+\d+\s+items\b", re.IGNORECASE),
    re.compile(r"={2,}\s*FAILURES\s*={2,}", re.IGNORECASE),
)
_UNKNOWN_DIFF_SIGNAL_PATTERNS = (
    re.compile(r"^diff --git", re.IGNORECASE | re.MULTILINE),
    re.compile(r"^(M|A|D|\?\?)\s", re.MULTILINE),
)


class AbstractAgent:
    def __init__(self, *args, **kwargs):
        model: AbstractModel
        replay_config: BaseModel | None
        logger: logging.Logger

    @classmethod
    def from_config(cls, config: AgentConfig) -> Self: ...

    def add_hook(self, hook: AbstractAgentHook) -> None: ...

    def get_trajectory_data(self) -> dict[str, Any]: ...

    def step(self) -> StepOutput: ...

    def run(self, *args, **kwargs) -> AgentRunResult: ...


def get_agent_from_config(config: AgentConfig) -> AbstractAgent:
    if config.type == "default":
        return DefaultAgent.from_config(config)
    elif config.type == "retry":
        return RetryAgent.from_config(config)
    elif config.type == "shell":
        # Need to defer import to avoid circular dependency
        from sweagent.agent.extra.shell_agent import ShellAgent

        return ShellAgent.from_config(config)
    else:
        msg = f"Unknown agent type: {config.type}"
        raise ValueError(msg)


class RetryAgent(AbstractAgent):
    def __init__(self, config: RetryAgentConfig):
        # Always copy config to avoid shared state between different instances
        self.config = config.model_copy(deep=True)
        self._hooks = []
        self._i_attempt = 0
        self.logger = get_logger("swea-agent", emoji="🤠")
        self._agent: DefaultAgent | None = None
        self._attempt_data: list[dict[str, Any]] = []
        self._total_instance_attempt_stats = InstanceStats()
        """Note that total_instance_attempt_stats only accumulates the states of the sub-agent,
        not the reviewer. Use self._total_instance_stats for the total stats.
        """
        self._chook = CombinedAgentHook()
        self._traj_path: Path | None = None
        self._problem_statement: ProblemStatement | None = None
        self._env: SWEEnv | None = None
        self._output_dir: Path | None = None
        self._rloop: ScoreRetryLoop | ChooserRetryLoop | None = None

    @property
    def _total_instance_stats(self) -> InstanceStats:
        assert self._rloop is not None
        return self._total_instance_attempt_stats + self._rloop.review_model_stats

    @classmethod
    def from_config(cls, config: RetryAgentConfig) -> Self:
        return cls(config)

    def add_hook(self, hook: AbstractAgentHook) -> None:
        self._chook.add_hook(hook)
        self._hooks.append(hook)

    def setup(
        self, env: SWEEnv, problem_statement: ProblemStatement | ProblemStatementConfig, output_dir: Path = Path(".")
    ) -> None:
        """Setup the retry agent for a new problem instance.
        This is mostly a bookkeeping step.
        """
        self._total_instance_attempt_stats = InstanceStats()
        self._problem_statement = problem_statement
        self._traj_path = output_dir / (self._problem_statement.id + ".traj")
        self._env = env
        self._output_dir = output_dir
        self._rloop = get_retry_loop_from_config(self.config.retry_loop, problem_statement=problem_statement)

    def _setup_agent(self) -> AbstractAgent:
        """Setup the agent for the current attempt."""
        # todo: Could select "best" agent config based on previous attempts if I run > number of set up configs
        agent_config = self.config.agent_configs[self._i_attempt % len(self.config.agent_configs)].model_copy(deep=True)
        remaining_budget = self.config.retry_loop.cost_limit - self._total_instance_stats.instance_cost
        if remaining_budget < agent_config.model.per_instance_cost_limit:
            self.logger.debug("Setting agent per-attempt cost limit to remaining budget: %s", remaining_budget)
            agent_config.model.per_instance_cost_limit = remaining_budget
        self._agent = DefaultAgent.from_config(agent_config)
        for hook in self._hooks:
            self._agent.add_hook(hook)
        assert self._output_dir is not None
        sub_agent_output_dir = self._output_dir / f"attempt_{self._i_attempt}"
        assert self._problem_statement is not None
        assert self._env is not None
        self._agent.setup(env=self._env, problem_statement=self._problem_statement, output_dir=sub_agent_output_dir)
        return self._agent

    def _next_attempt(self) -> None:
        """Prepare for the next attempt: Reset the environment and setup the next agent."""
        assert self._env is not None
        self._i_attempt += 1
        self._env.hard_reset()
        self._setup_agent()

    def step(self) -> StepOutput:
        """Step the agent of the current attempt.
        Attempt autosubmit if an error occurs (though all errors should already be handled by the attempt agent).
        """
        assert self._agent is not None
        # Failsafe cost check, this should not actually happen, because the sub-agent should have already been
        # initialized with the correct cost limit to not exceed the total cost limit. Using factor of 1.1, because
        # sub-agent might only catch the cost limit after attempting.
        if self._total_instance_stats.instance_cost > 1.1 * self.config.retry_loop.cost_limit > 0:
            msg = "Total instance cost exceeded cost limit. This should not happen, please report this. Triggering autosubmit."
            self.logger.critical(msg)
            return self._agent.attempt_autosubmission_after_error(step=StepOutput())
        try:
            step = self._agent.step()
        except TotalCostLimitExceededError:
            # Need to make sure that this error causes everything to stop
            raise
        except Exception as e:
            msg = "Error in agent step: %s. This really shouldn't happen, please report this. Triggering autosubmit."
            self.logger.critical(msg, e, exc_info=True)
            step = self._agent.attempt_autosubmission_after_error(step=StepOutput())
        return step

    def _finalize_agent_run(self) -> None:
        """Add the agent results to our list of results"""
        assert self._agent is not None
        self._agent.save_trajectory()
        self._attempt_data.append(self._agent.get_trajectory_data())
        self._total_instance_attempt_stats += self._agent.model.stats

    def get_trajectory_data(self, choose: bool) -> dict[str, Any]:
        """Get all data that we save in .traj files."""
        assert self._rloop is not None

        data = {
            "attempts": self._attempt_data,
        }

        if choose:
            try:
                best_attempt_idx = self._rloop.get_best()
            except TotalCostLimitExceededError:
                raise
            except Exception as e:
                self.logger.critical(f"Error getting best attempt index: {e}. Setting to 0.", exc_info=True)
                best_attempt_idx = 0
            data |= copy.deepcopy(self._attempt_data[best_attempt_idx])  # type: ignore
            data["info"]["best_attempt_idx"] = best_attempt_idx
            data["info"]["rloop_model_stats"] = self._rloop.review_model_stats.model_dump()
            # Overwrite model stats with total stats
            data["info"]["model_stats"] = self._total_instance_stats.model_dump()
            if isinstance(self._rloop, ChooserRetryLoop):
                data["info"]["chooser"] = (
                    self._rloop._chooser_output.model_dump() if self._rloop._chooser_output else {}
                )
        return data

    def save_trajectory(self, choose: bool) -> None:
        data = self.get_trajectory_data(choose=choose)
        assert self._traj_path is not None
        self._traj_path.write_text(json.dumps(data, indent=2))

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance. This method contains the
        main loop that repeatedly calls `self._step` until the problem is solved.

        Args:
            env: The environment to run the agent on.
            problem_statement: The problem statement to run the agent on.
            output_dir: Directory to save the trajectory to
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)
        assert self._rloop is not None

        # Run action/observation loop
        self._chook.on_run_start()
        step_output = StepOutput()
        self._setup_agent()
        assert self._agent is not None
        while not step_output.done:
            step_output = self.step()
            self.save_trajectory(choose=False)
            if step_output.done:
                self._rloop.on_submit(
                    ReviewSubmission(
                        trajectory=self._agent.trajectory,
                        info=self._agent.info,
                        model_stats=self._agent.model.stats,
                    )
                )
                if isinstance(self._rloop, ScoreRetryLoop):
                    self._agent.info["review"] = self._rloop.reviews[-1].model_dump()  # type: ignore
                self._finalize_agent_run()
                self.save_trajectory(choose=False)
                if self._rloop.retry():
                    assert self._env is not None
                    self._next_attempt()
                    step_output.done = False
        self.save_trajectory(choose=True)  # call again after we finalized
        self._chook.on_run_done(trajectory=self._agent.trajectory, info=self._agent.info)

        self.logger.info("Trajectory saved to %s", self._traj_path)

        # Here we want to return the "global" information (e.g., submission should
        # be the best submission instead of the last one, etc.), so we get it from the traj file
        data = self.get_trajectory_data(choose=True)
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])


class DefaultAgent(AbstractAgent):
    def __init__(
        self,
        *,
        templates: TemplateConfig,
        tools: ToolHandler,
        history_processors: list[HistoryProcessor],
        model: AbstractModel,
        max_requeries: int = 3,
        name: str = "main",
        _catch_errors: bool = True,
        _always_require_zero_exit_code: bool = False,
        action_sampler_config: ActionSamplerConfig | None = None,
    ):
        """The agent handles the behaviour of the model and how it interacts with the environment.

        To run the agent, either call `self.run` or `self.setup` and then `self.step` in a loop.
        """
        self._catch_errors = _catch_errors
        self._always_require_zero_exit_code = _always_require_zero_exit_code
        self.name = name
        self.model = model
        self.templates = templates
        self.tools = tools
        if isinstance(self.model, HumanThoughtModel):
            self.tools.config.parse_function = ThoughtActionParser()
        elif isinstance(self.model, HumanModel):
            self.tools.config.parse_function = ActionOnlyParser()
        self.history_processors = history_processors
        self.max_requeries = max_requeries
        self.logger = get_logger("swea-agent", emoji="🤠")
        # Set in run method
        self._env: SWEEnv | None = None
        self._problem_statement: ProblemStatement | ProblemStatementConfig | None = None
        self.traj_path: Path | None = None

        #: The following three attributes collect the information about how the agent
        #: solved the problem.
        self.history = []
        self._trajectory = []
        self.info = AgentInfo()

        self._chook = CombinedAgentHook()

        self._replay_config: BaseModel | None = None
        """This can be set to a RunSingleConfig from the Run instance whenever possible.
        It can be used to replay the agent's trajectory in an environment.
        """

        self._action_sampler: AbstractActionSampler | None = None
        if action_sampler_config is not None:
            self._action_sampler = action_sampler_config.get(self.model, self.tools)

        #: Count how many timeout errors have occurred consecutively. Kills agent
        #: after 5 of them.
        self._n_consecutive_timeouts = 0
        self._total_execution_time = 0.0
        self._thermo_stats: dict[str, int] = {"useful_work": 0, "friction_loss": 0, "unknown": 0}
        self._init_probe_state()

    def _init_probe_state(self) -> None:
        """Initialize / reset all multi-dimensional probe accumulators."""
        # action_taxonomy: counts per action category
        self._action_counts: dict[str, int] = {}
        self._action_step_count: int = 0
        self._action_total_commands: int = 0
        # token_efficiency: per-step token deltas (list of (input_delta, output_delta))
        self._token_snapshots: list[tuple[int, int]] = []
        self._prev_tokens_sent: int = 0
        self._prev_tokens_received: int = 0
        self._token_zero_delta_steps: int = 0
        self._token_useful_total: int = 0
        # observation_bloat: observation size buckets
        self._obs_size_buckets: dict[str, int] = {"small": 0, "medium": 0, "large": 0, "huge": 0}
        self._obs_total_chars: int = 0
        self._obs_max_chars: int = 0
        # iteration_waste: repeated edits on same file, reverts
        self._edit_file_history: list[str] = []
        self._repeated_edits: int = 0
        self._reverts: int = 0
        self._duplicate_actions: int = 0
        self._duplicate_observations: int = 0
        self._last_action_signature: str = ""
        self._last_observation_signature: str = ""
        self._last_edited_file: str | None = None
        # redundancy_verification: fine-grained redundancy tracking
        self._submitted: bool = False
        self._post_submit_steps: int = 0
        self._post_submit_tokens: int = 0
        self._file_read_history: dict[str, list[int]] = {}  # path -> [step_indices]
        self._reread_immediate: int = 0
        self._reread_after_edit: int = 0
        self._reread_distant: int = 0
        self._reread_tokens: int = 0
        self._git_check_count: int = 0
        self._repeated_git_checks: int = 0
        self._repeated_git_check_tokens: int = 0
        self._test_history: list[tuple[str, str]] = []  # (command_sig, obs_hash)
        self._duplicate_test_same_result: int = 0
        self._duplicate_test_same_result_tokens: int = 0
        self._retest_different_result: int = 0
        self._retest_different_result_tokens: int = 0
        self._python_run_history: list[tuple[str, str]] = []  # (command_sig, obs_hash)
        self._duplicate_python_verify: int = 0
        self._duplicate_python_verify_tokens: int = 0
        self._step_index: int = 0
        self._step_edit_ranges: dict[str, list[int]] = {}  # path -> [step_indices where edited]
        # context_pressure: track query (prompt) size growth
        self._query_sizes: list[int] = []
        self._query_size_deltas: list[int] = []
        # unknown_diagnostics: explain why thermo classification landed in "unknown"
        self._unknown_steps: int = 0
        self._unknown_reason_counts: dict[str, int] = {}
        self._unknown_error_signal_steps: int = 0
        self._unknown_test_signal_steps: int = 0
        self._unknown_long_observation_steps: int = 0

    @classmethod
    def from_config(cls, config: DefaultAgentConfig) -> Self:
        # To ensure that all models stay completely independent, we deepcopy the
        # model config, because it lives on as a property in the model, tools, etc.
        config = config.model_copy(deep=True)
        model = get_model(config.model, config.tools)
        return cls(
            templates=config.templates,
            tools=ToolHandler(config.tools),
            history_processors=config.history_processors,
            model=model,
            max_requeries=config.max_requeries,
            action_sampler_config=config.action_sampler,
        )

    def add_hook(self, hook: AbstractAgentHook) -> None:
        """Add hook to agent"""
        hook.on_init(agent=self)
        self._chook.add_hook(hook)

    # Properties
    # ----------

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    @property
    def replay_config(self) -> BaseModel | None:
        return self._replay_config

    @replay_config.setter
    def replay_config(self, value: BaseModel):
        # Do import here to avoid circular dependency
        from sweagent.run.run_single import RunSingleConfig

        self._replay_config = RunSingleConfig.model_validate(_strip_abspath_from_dict(value.model_dump()))

    @property
    def messages(self) -> list[dict[str, Any]]:
        """Return the history of the agent for this attempt since the last reset,
        processed through all history processors.
        """
        filtered_history = [entry for entry in self.history if entry["agent"] == self.name]  # type: ignore

        # Chain the history processors
        messages = filtered_history
        for processor in self.history_processors:
            messages = processor(messages)

        return messages  # type: ignore

    # Methods
    # -------

    def _append_history(self, item: dict[str, Any]) -> None:
        """Adds an item to the history."""
        self._chook.on_query_message_added(**item)
        self.history.append(item)  # type: ignore

    def setup(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> None:
        """Setup the agent for a new instance. This includes
        formatting the system message and adding demonstrations to the history.

        This method is called by `self.run`.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # apply template configuration to multimodal problem statements
        if hasattr(problem_statement, "type") and problem_statement.type == "swe_bench_multimodal":
            from sweagent.agent.problem_statement import SWEBenchMultimodalProblemStatement

            if isinstance(problem_statement, SWEBenchMultimodalProblemStatement):
                # apply the global disable_image_processing setting if it's not explicitly set
                if not problem_statement.disable_image_processing and self.templates.disable_image_processing:
                    problem_statement.disable_image_processing = True

        self._problem_statement = problem_statement
        self._env = env
        iid = self._problem_statement.id
        self.logger.info("Setting up agent for instance %s", iid)

        # Save/reset some attributes
        self.traj_path = output_dir / (self._problem_statement.id + ".traj")
        self.logger.info("Trajectory will be saved to %s", self.traj_path)

        self._chook.on_tools_installation_started()
        self.tools.install(self._env)
        self._chook.on_setup_attempt()
        self.info = AgentInfo()
        self.info["swe_agent_hash"] = get_agent_commit_hash()
        self.info["swe_agent_version"] = __version__
        self.info["swe_rex_version"] = get_rex_version()
        self.info["swe_rex_hash"] = get_rex_commit_hash()
        self._thermo_stats = {"useful_work": 0, "friction_loss": 0, "unknown": 0}
        self._init_probe_state()
        assert self._env is not None
        assert self._problem_statement is not None
        self._env.set_env_variables({"PROBLEM_STATEMENT": self._problem_statement.get_problem_statement_for_env()})
        self.add_system_message_to_history()
        self.add_demonstrations_to_history()
        self.add_instance_template_to_history(state=self.tools.get_state(self._env))
        self._chook.on_setup_done()

    def add_system_message_to_history(self) -> None:
        """Add system message to history"""
        assert self._problem_statement is not None
        system_msg = Template(self.templates.system_template).render(**self._get_format_dict())
        self.logger.info(f"SYSTEM ({self.name})\n{system_msg}")
        self._append_history(
            {"role": "system", "content": system_msg, "agent": self.name, "message_type": "system_prompt"}
        )

    def add_demonstrations_to_history(self) -> None:
        """Add demonstrations to history"""
        for demonstration_path in self.templates.demonstrations:
            self._add_demonstration_to_history(demonstration_path)

    def _add_demonstration_to_history(self, demonstration_path: Path) -> None:
        """Load demonstration from disk and add to history"""
        if self.templates.demonstration_template is None and not self.templates.put_demos_in_history:
            msg = "Cannot use demonstrations without a demonstration template or put_demos_in_history=True"
            raise ValueError(msg)

        # Load history
        self.logger.info(f"DEMONSTRATION: {demonstration_path}")
        _demo_text = Path(demonstration_path).read_text()
        if demonstration_path.suffix == ".yaml":
            demo_history = yaml.safe_load(_demo_text)["history"]
        else:
            demo_history = json.loads(_demo_text)["history"]

        if self.templates.put_demos_in_history:
            # Add demonstrations to history step-by-step
            for entry in demo_history:
                if entry["role"] != "system":
                    entry["is_demo"] = True
                    self._append_history(entry)
        else:
            # Add demonstration as single message to history
            demo_history = [entry for entry in demo_history if entry["role"] != "system"]
            demo_message = "\n".join([entry["content"] for entry in demo_history])
            assert self.templates.demonstration_template is not None
            demonstration = Template(self.templates.demonstration_template).render(demonstration=demo_message)
            self._append_history(
                {
                    "agent": self.name,
                    "content": demonstration,
                    "is_demo": True,
                    "role": "user",
                    "message_type": "demonstration",
                },
            )

    def _get_format_dict(self, **kwargs) -> dict[str, Any]:
        """Get the dictionary of key value pairs used to format the templates

        Args:
            **kwargs: additional keyword arguments to be added to the format dictionary
        """
        assert self._problem_statement is not None
        assert self._env is not None
        return dict(
            command_docs=self.tools.config.command_docs,
            **self.tools.config.env_variables,
            **kwargs,
            problem_statement=self._problem_statement.get_problem_statement(),
            repo=self._env.repo.repo_name if self._env.repo is not None else "",
            **self._problem_statement.get_extra_fields(),
        )

    def _add_templated_messages_to_history(
        self, templates: list[str], tool_call_ids: list[str] | None = None, **kwargs: str | int | None
    ) -> None:
        """Populate selected template(s) with information (e.g., issue, arguments, state)
        and add to history.

        Args:
            templates: templates to populate and add to history
            tool_call_ids: tool call ids to be added to the history
            **kwargs: keyword arguments to be passed to the templates (in addition to the
                ones in `self._get_format_dict`)
        """
        messages = []

        format_dict = self._get_format_dict(**kwargs)
        for template in templates:
            try:
                messages.append(Template(template).render(**format_dict))
            except KeyError:
                self.logger.debug("The following keys are available: %s", format_dict.keys())
                raise

        message = "\n".join(messages)

        # We disable syntax highlighting here, because some inputs can lead to a complete cross-thread
        # freeze in the agent. See https://github.com/SWE-agent/SWE-agent/issues/901 .
        self.logger.info(f"🤖 MODEL INPUT\n{message}", extra={"highlighter": None})
        history_item: dict[str, Any] = {
            "role": "user",
            "content": message,
            "agent": self.name,
            "message_type": "observation",
        }
        if tool_call_ids:
            assert len(tool_call_ids) == 1, "This should be ensured by the FunctionCalling parse method"
            history_item["role"] = "tool"
            history_item["tool_call_ids"] = tool_call_ids
        self._append_history(history_item)

    def add_step_to_history(self, step: StepOutput) -> None:
        """Adds a step (command that was run and output) to the model history"""
        self._append_history(
            {
                "role": "assistant",
                "content": step.output,
                "thought": step.thought,
                "action": step.action,
                "agent": self.name,
                "tool_calls": step.tool_calls,
                "message_type": "action",
                "thinking_blocks": step.thinking_blocks,
            },
        )

        elided_chars = 0
        if step.observation.strip() == "":
            # Show no output template if observation content was empty
            templates = [self.templates.next_step_no_output_template]
        elif len(step.observation) > self.templates.max_observation_length:
            templates = [self.templates.next_step_truncated_observation_template]
            elided_chars = len(step.observation) - self.templates.max_observation_length
        else:
            # Show standard output template if there is observation content
            templates = [self.templates.next_step_template]
        self._add_templated_messages_to_history(
            templates,
            observation=step.observation,
            elided_chars=elided_chars,
            max_observation_length=self.templates.max_observation_length,
            tool_call_ids=step.tool_call_ids,
            **step.state,
        )

    def add_instance_template_to_history(self, state: dict[str, str]) -> None:
        """Add observation to history, as well as the instance template or demonstrations if we're
        at the start of a new attempt.
        """
        templates: list[str] = []
        # Determine observation template based on what prior observation was
        assert self.history[-1]["role"] == "system" or self.history[-1].get("is_demo", False)
        # Show instance template if prev. obs. was initial system message
        templates = [self.templates.instance_template]
        if self.templates.strategy_template is not None:
            templates.append(self.templates.strategy_template)

        self._add_templated_messages_to_history(templates, **state)  # type: ignore

    def get_trajectory_data(self) -> dict[str, Any]:
        """Get all data that we save in .traj files."""

        assert self._env is not None
        # The deepcopy here is important because else the
        # data["info"]["model_stats"] update will create havoc!
        attempt_data = copy.deepcopy(
            {
                "trajectory": self.trajectory,
                "history": self.history,
                "info": self.info,
            }
        )
        attempt_data["replay_config"] = self.replay_config.model_dump_json() if self.replay_config is not None else None
        attempt_data["environment"] = self._env.name
        return attempt_data

    def save_trajectory(
        self,
    ) -> None:
        """Save the trajectory to disk.
        This includes the history, the environment state, and the model stats.
        """
        data = self.get_trajectory_data()
        assert self.traj_path is not None
        self.traj_path.write_text(json.dumps(data, indent=2))

    def get_model_requery_history(
        self, error_template: str, *, output: str, **kwargs: str | int | float | bool | None
    ) -> list[dict[str, str]]:
        """Ask the model to correct after a hitting one of the following errors:

        1. Malformatted output (could not parse action)
        2. Blocked action (command is on the blocklist)
        3. Bash command syntax error

        At the time this function is called, the proposed action and observation are not part of the history
        yet.

        This function adds temporary history based on the error template and queries the model.
        If the model is able to correct itself, the records of the mistakes will not be part of the history
        (but they are saved in the trajectory).

        Args:
            error_template: error template
            output: model output
            **kwargs: keyword arguments to be passed to the error template

        Returns:
            model output after requery
        """
        format_dict = {**kwargs, **self._get_format_dict()}
        error_template = Template(error_template).render(**format_dict)

        self.logger.warning(f"{error_template}")

        return self.messages + [
            {"role": "assistant", "content": output, "agent": self.name, "message_type": "assistant"},
            {"role": "user", "content": error_template, "agent": self.name, "message_type": "user"},
        ]

    def attempt_autosubmission_after_error(self, step: StepOutput) -> StepOutput:
        """For most exceptions, we attempt to still extract the patch and submit that.
        This means we send the `submit` command to the runtime and parse the output.
        """
        self.logger.warning("Attempting autosubmission after error")
        step = step.model_copy(deep=True)
        step.done = True
        assert self._env is not None
        if not asyncio.run(self._env.deployment.is_alive(timeout=10)):
            # The agent is dead. This is very bad. Maybe we can take a 'diff' that was saved
            # for a previous step? (if running with diff in tools)
            self.logger.error("Runtime is no longer alive")
            try:
                last_trajectory_step = self.trajectory[-1]
            except IndexError:
                self.logger.info("No last trajectory step to extract patch from")
                return step
            if "diff" not in last_trajectory_step["state"]:
                self.logger.info("No diff in last trajectory step state, cannot autosubmit")
                return step
            diff = last_trajectory_step["state"]["diff"]
            self.logger.info("Using diff from last trajectory step to autosubmit")
            step.submission = diff
            if step.submission:
                step.observation = "Environment died unexpectedly. Exited (autosubmitted)"
                step.exit_status = f"submitted ({step.exit_status})"
            else:
                self.logger.info("Diff from last traj step empty.")
            return step
        # Let us manually run the submission command and collect the output
        repo_name = "/"
        if self._env.repo is not None:
            repo_name = runtime_repo_path(self._env.repo.repo_name)
        patch_path = _model_patch_path()
        submission_command = f"git add -A && git diff --cached > {patch_path}"
        self.logger.info("Executing submission command %s in %s", submission_command, repo_name)
        try:
            self._env.execute_command(submission_command, check=True, cwd=repo_name)
        except Exception as e:
            self.logger.error("Failed to execute submission command, got %s", e)
        # There's still hope for the submission, because the `/root/model.patch` file might have been
        # generated by the state command
        step = self.handle_submission(step, observation="", force_submission=True)
        if step.submission:
            self.logger.info("Exiting with autosubmission")
            step.observation = "Exited (autosubmitted)"
        return step

    def handle_submission(self, step: StepOutput, *, observation="", force_submission: bool = False) -> StepOutput:
        """Check if there was a submission in the observation and handle it.

        Args:
            step:
            observation: If specified, will use this rather than stepobservation
            force_submission: If True, will always submit even if no submission is found

        Returns:
            step: step with submission and observation updated (if submission was found)
        """
        step = step.model_copy(deep=True)
        assert self.tools is not None
        is_submission = self.tools.check_for_submission_cmd(observation or step.observation)
        if is_submission or force_submission:
            assert self._env is not None
            submission = None
            patch_candidates = [_model_patch_path(), "/root/model.patch"]
            for patch_path in patch_candidates:
                try:
                    submission = self._env.read_file(patch_path, encoding="utf-8", errors="backslashreplace")
                    break
                except FileNotFoundError:
                    continue
                except Exception as e:
                    self.logger.exception("Failed to read submission file %s, got %s", patch_path, e)
                    return step
            if submission is None:
                self.logger.warning("Submission file not found, no submission was made")
                return step
            if submission.strip() != "":
                step.submission = submission
            else:
                step.submission = None
            step.observation = submission
            if not step.exit_status:
                step.exit_status = "submitted"
            elif step.submission:
                step.exit_status = f"submitted ({step.exit_status})"
            step.done = True
            self.logger.info(f"Found submission: {submission}")
        return step

    def _get_edited_files_with_context(self, patch: str) -> dict[str, str]:
        """Get the edited files with context from the patch"""
        assert self._env is not None
        try:
            if self._env.repo is None:
                pf = None
            else:
                pf = (
                    PatchFormatter(
                        patch,
                        read_method=lambda path: self._env.read_file(  # type: ignore[attr-defined]
                            PurePosixPath(runtime_repo_path(self._env.repo.repo_name)) / path  # type: ignore[attr-defined]
                        ),
                    )
                    if patch
                    else None
                )
        except UnidiffParseError:
            self.logger.error("Failed to parse patch with unidiff. Some variables will be empty.")
            pf = None
            # We still need to populate the variables
        out = {}
        for context_length in [30, 50, 70]:
            value = "Empty. No edited files found."
            if pf is not None:
                value = pf.get_files_str(original=False, context_length=context_length)
            out[f"edited_files{context_length}"] = value
        return out

    def _classify_thermo_outcome(self, step: StepOutput) -> str:
        observation = (step.observation or "").lower()
        state = step.state if isinstance(step.state, dict) else {}
        diff_text = str(state.get("diff", "")).strip()
        if diff_text or (step.submission or "").strip():
            return "useful_work"
        if any(re.search(pattern, observation, re.IGNORECASE) for pattern in THERMO_USEFUL_PATTERNS):
            return "useful_work"
        if any(re.search(pattern, observation, re.IGNORECASE) for pattern in THERMO_FRICTION_PATTERNS):
            return "friction_loss"
        return "unknown"

    def _record_thermo_probe(self, step: StepOutput) -> None:
        label = self._classify_thermo_outcome(step)
        self._thermo_stats[label] = self._thermo_stats.get(label, 0) + 1
        total = sum(self._thermo_stats.values())
        friction = self._thermo_stats.get("friction_loss", 0)
        step.extra_info = step.extra_info or {}
        step.extra_info["thermo_probe"] = {
            "classification": label,
            "useful_work": self._thermo_stats.get("useful_work", 0),
            "friction_loss": friction,
            "unknown": self._thermo_stats.get("unknown", 0),
            "friction_ratio": (friction / total) if total else 0.0,
        }

    @staticmethod
    def _safe_content_size(content: Any) -> int:
        if content is None:
            return 0
        if isinstance(content, str):
            return len(content)
        try:
            return len(json.dumps(content))
        except TypeError:
            return len(str(content))

    def _estimate_query_size(self, query: list[dict[str, Any]]) -> int:
        total = 0
        for item in query:
            if not isinstance(item, dict):
                total += self._safe_content_size(item)
                continue
            total += self._safe_content_size(item.get("role"))
            total += self._safe_content_size(item.get("message_type"))
            total += self._safe_content_size(item.get("content"))
            total += self._safe_content_size(item.get("tool_calls"))
            total += self._safe_content_size(item.get("thinking_blocks"))
        return total

    def _split_action_commands(self, action: str) -> list[str]:
        commands: list[str] = []
        for line in (action or "").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            pieces = re.split(r"\s*(?:&&|\|\||;)\s*", stripped)
            commands.extend(piece for piece in (p.strip() for p in pieces) if piece)
        return commands

    def _classify_action_command(self, command: str) -> str:
        cmd = command.strip()
        for label, pattern in _ACTION_TAXONOMY:
            if pattern.search(cmd):
                return label
        return "other"

    def _extract_edit_target(self, command: str) -> str | None:
        try:
            tokens = shlex.split(command)
        except ValueError:
            tokens = command.split()
        if not tokens:
            return None
        if tokens[0] == "str_replace_editor" and len(tokens) >= 3 and tokens[1] in {"str_replace", "create", "insert"}:
            return tokens[2]
        if tokens[0] in {"sed", "patch", "edit"} and len(tokens) >= 2:
            return tokens[-1]
        return None

    @staticmethod
    def _median(values: list[int]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        mid = len(sorted_values) // 2
        if len(sorted_values) % 2 == 1:
            return float(sorted_values[mid])
        return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0

    def _record_action_taxonomy_probe(self, commands: list[str], categories: list[str]) -> dict[str, Any]:
        self._action_step_count += 1
        step_counts: dict[str, int] = {}
        if not commands:
            self._action_counts["no_action"] = self._action_counts.get("no_action", 0) + 1
            step_counts["no_action"] = 1
            return {
                "primary_action": "no_action",
                "command_count": 0,
                "categories": [],
                "step_counts": step_counts,
            }
        self._action_total_commands += len(commands)
        for category in categories:
            self._action_counts[category] = self._action_counts.get(category, 0) + 1
            step_counts[category] = step_counts.get(category, 0) + 1
        return {
            "primary_action": categories[0],
            "command_count": len(commands),
            "categories": categories,
            "step_counts": step_counts,
        }

    def _record_token_efficiency_probe(self, step: StepOutput) -> dict[str, Any]:
        tokens_sent = int(self.model.stats.tokens_sent)
        tokens_received = int(self.model.stats.tokens_received)
        delta_in = max(tokens_sent - self._prev_tokens_sent, 0)
        delta_out = max(tokens_received - self._prev_tokens_received, 0)
        delta_total = delta_in + delta_out
        self._prev_tokens_sent = tokens_sent
        self._prev_tokens_received = tokens_received
        self._token_snapshots.append((delta_in, delta_out))
        if delta_total == 0:
            self._token_zero_delta_steps += 1
        thermo_classification = (step.extra_info or {}).get("thermo_probe", {}).get("classification")
        if thermo_classification == "useful_work":
            self._token_useful_total += delta_total
        return {
            "delta_input_tokens": delta_in,
            "delta_output_tokens": delta_out,
            "delta_total_tokens": delta_total,
            "cumulative_input_tokens": tokens_sent,
            "cumulative_output_tokens": tokens_received,
            "cumulative_total_tokens": tokens_sent + tokens_received,
        }

    def _record_observation_bloat_probe(self, step: StepOutput) -> dict[str, Any]:
        obs_chars = len(step.observation or "")
        self._obs_total_chars += obs_chars
        self._obs_max_chars = max(self._obs_max_chars, obs_chars)
        if obs_chars >= _OBS_HUGE_THRESHOLD:
            bucket = "huge"
        elif obs_chars >= _OBS_BLOAT_THRESHOLD:
            bucket = "large"
        elif obs_chars >= _OBS_MEDIUM_THRESHOLD:
            bucket = "medium"
        else:
            bucket = "small"
        self._obs_size_buckets[bucket] = self._obs_size_buckets.get(bucket, 0) + 1
        return {
            "observation_chars": obs_chars,
            "bucket": bucket,
            "is_large": bucket in {"large", "huge"},
            "is_huge": bucket == "huge",
        }

    def _record_iteration_waste_probe(self, step: StepOutput, commands: list[str], categories: list[str]) -> dict[str, Any]:
        repeated_action = False
        repeated_observation = False
        repeated_edit = False

        # Estimate token delta for this step
        te_info = (step.extra_info or {}).get("token_efficiency", {})
        step_delta_tokens = te_info.get("delta_input_tokens", 0) or 0

        action_signature = " ".join((step.action or "").split())
        if action_signature and action_signature == self._last_action_signature:
            repeated_action = True
            self._duplicate_actions += 1
        if action_signature:
            self._last_action_signature = action_signature

        observation = (step.observation or "").strip()
        observation_signature = (
            hashlib.sha1(observation.encode("utf-8", errors="ignore")).hexdigest() if observation else ""
        )
        if observation_signature and observation_signature == self._last_observation_signature:
            repeated_observation = True
            self._duplicate_observations += 1
        if observation_signature:
            self._last_observation_signature = observation_signature

        edited_targets: list[str] = []
        for command, category in zip(commands, categories):
            if category != "edit":
                continue
            target = self._extract_edit_target(command)
            if not target:
                continue
            edited_targets.append(target)
            self._edit_file_history.append(target)
            if self._last_edited_file == target:
                repeated_edit = True
                self._repeated_edits += 1
            self._last_edited_file = target
            # Track edit ranges for reread classification
            self._step_edit_ranges.setdefault(target, []).append(self._step_index)
        if len(self._edit_file_history) > 500:
            self._edit_file_history = self._edit_file_history[-500:]

        revert_hits = 0
        for command in commands:
            if any(pattern.search(command) for pattern in _ITERATION_WASTE_PATTERNS):
                revert_hits += 1
        self._reverts += revert_hits

        # ── Fine-grained redundancy verification ────────────────────────
        is_post_submit = False
        reread_type: str | None = None
        is_repeated_git_check = False
        test_redundancy: str | None = None      # "duplicate_same_result" | "retest_different_result" | None
        python_redundancy: str | None = None    # "duplicate_same_result" | None

        action_str = (step.action or "").strip()

        # 1. Post-submit tracking
        if action_str.startswith("submit"):
            self._submitted = True
        elif self._submitted and action_str:
            is_post_submit = True
            self._post_submit_steps += 1
            self._post_submit_tokens += step_delta_tokens

        # 2. File re-read detection with reason classification
        for command in commands:
            m = _FILE_READ_PATTERN.search(command)
            if not m:
                continue
            fpath = m.group(1) or m.group(2)
            if not fpath or len(fpath) < 3:
                continue
            if fpath in self._file_read_history:
                prev_steps = self._file_read_history[fpath]
                last_read = prev_steps[-1]
                gap = self._step_index - last_read
                # Was the file edited between the last read and now?
                edit_steps = self._step_edit_ranges.get(fpath, [])
                edited_between = any(last_read < es < self._step_index for es in edit_steps)
                if edited_between:
                    self._reread_after_edit += 1
                    reread_type = "after_edit"
                elif gap <= 2:
                    self._reread_immediate += 1
                    reread_type = "immediate"
                else:
                    self._reread_distant += 1
                    reread_type = "distant"
                self._reread_tokens += step_delta_tokens
            self._file_read_history.setdefault(fpath, []).append(self._step_index)

        # 3. Repeated git check (git diff/status/log/show)
        for command in commands:
            if _GIT_CHECK_PATTERN.search(command):
                self._git_check_count += 1
                if self._git_check_count > 1:
                    is_repeated_git_check = True
                    self._repeated_git_checks += 1
                    self._repeated_git_check_tokens += step_delta_tokens

        # 4. Test command redundancy
        for command in commands:
            if not _TEST_COMMAND_PATTERN.search(command):
                continue
            cmd_sig = command[:100]
            obs_hash = hashlib.sha1(observation[:500].encode("utf-8", errors="ignore")).hexdigest()
            matched_cmd = [t for t in self._test_history if t[0] == cmd_sig]
            if any(t[1] == obs_hash for t in matched_cmd):
                test_redundancy = "duplicate_same_result"
                self._duplicate_test_same_result += 1
                self._duplicate_test_same_result_tokens += step_delta_tokens
            elif matched_cmd:
                test_redundancy = "retest_different_result"
                self._retest_different_result += 1
                self._retest_different_result_tokens += step_delta_tokens
            self._test_history.append((cmd_sig, obs_hash))

        # 5. Python run redundancy (exclude test commands)
        for command in commands:
            if not _PYTHON_RUN_PATTERN.search(command):
                continue
            if _TEST_COMMAND_PATTERN.search(command):
                continue
            cmd_sig = command[:100]
            obs_hash = hashlib.sha1(observation[:500].encode("utf-8", errors="ignore")).hexdigest()
            matched_cmd = [p for p in self._python_run_history if p[0] == cmd_sig]
            if any(p[1] == obs_hash for p in matched_cmd):
                python_redundancy = "duplicate_same_result"
                self._duplicate_python_verify += 1
                self._duplicate_python_verify_tokens += step_delta_tokens
            self._python_run_history.append((cmd_sig, obs_hash))

        self._step_index += 1

        waste_events = int(repeated_action) + int(repeated_observation) + int(repeated_edit) + revert_hits
        return {
            "repeated_action": repeated_action,
            "repeated_observation": repeated_observation,
            "repeated_edit": repeated_edit,
            "revert_hits": revert_hits,
            "edited_targets": edited_targets,
            "waste_events": waste_events,
            "redundancy": {
                "is_post_submit": is_post_submit,
                "reread_type": reread_type,
                "is_repeated_git_check": is_repeated_git_check,
                "test_redundancy": test_redundancy,
                "python_redundancy": python_redundancy,
            },
        }

    def _record_context_pressure_probe(self, step: StepOutput) -> dict[str, Any]:
        query_size = self._estimate_query_size(step.query)
        if self._query_sizes:
            delta = query_size - self._query_sizes[-1]
        else:
            delta = 0
        self._query_sizes.append(query_size)
        self._query_size_deltas.append(delta)
        baseline = self._query_sizes[0] if self._query_sizes else 0
        pressure_index = (query_size / baseline) if baseline else 0.0
        return {
            "query_chars": query_size,
            "delta_query_chars": delta,
            "pressure_index": pressure_index,
        }

    def _record_unknown_diagnostics_probe(self, step: StepOutput, commands: list[str], categories: list[str]) -> dict[str, Any]:
        step.extra_info = step.extra_info or {}
        thermo_label = (step.extra_info.get("thermo_probe") or {}).get("classification", "")
        observation = step.observation or ""
        observation_chars = len(observation)
        has_error_signal = any(pattern.search(observation) for pattern in _UNKNOWN_ERROR_SIGNAL_PATTERNS)
        has_test_signal = any(pattern.search(observation) for pattern in _UNKNOWN_TEST_SIGNAL_PATTERNS)
        has_diff_signal = any(pattern.search(observation) for pattern in _UNKNOWN_DIFF_SIGNAL_PATTERNS)
        categories_set = set(categories)

        reason = "not_unknown"
        if thermo_label == "unknown":
            self._unknown_steps += 1
            if has_error_signal:
                self._unknown_error_signal_steps += 1
            if has_test_signal:
                self._unknown_test_signal_steps += 1
            if observation_chars >= _OBS_HUGE_THRESHOLD:
                self._unknown_long_observation_steps += 1

            has_execution = bool(categories_set.intersection({"edit", "python", "test", "cleanup", "submit", "exit"}))
            has_exploration = bool(categories_set.intersection({"navigate", "search", "read_file", "git"}))

            if not commands:
                reason = "empty_action"
            elif has_exploration and not has_execution:
                reason = "exploration_or_navigation"
            elif categories_set.intersection({"python", "test"}) and has_error_signal and has_test_signal:
                reason = "test_failure_unmatched"
            elif categories_set.intersection({"python", "test"}) and has_error_signal:
                reason = "execution_error_unmatched"
            elif "edit" in categories_set and not has_diff_signal:
                reason = "edit_without_progress_signal"
            elif observation_chars >= _OBS_HUGE_THRESHOLD:
                reason = "long_observation_neutral"
            else:
                reason = "uncategorized_unknown"

            self._unknown_reason_counts[reason] = self._unknown_reason_counts.get(reason, 0) + 1

        return {
            "active": thermo_label == "unknown",
            "reason": reason,
            "has_error_signal": has_error_signal,
            "has_test_signal": has_test_signal,
            "has_diff_signal": has_diff_signal,
            "observation_chars": observation_chars,
        }

    def _record_multidim_probes(self, step: StepOutput) -> None:
        step.extra_info = step.extra_info or {}
        commands = self._split_action_commands(step.action)
        categories = [self._classify_action_command(command) for command in commands]
        step.extra_info["action_taxonomy"] = self._record_action_taxonomy_probe(commands, categories)
        step.extra_info["token_efficiency"] = self._record_token_efficiency_probe(step)
        step.extra_info["observation_bloat"] = self._record_observation_bloat_probe(step)
        step.extra_info["iteration_waste"] = self._record_iteration_waste_probe(step, commands, categories)
        step.extra_info["context_pressure"] = self._record_context_pressure_probe(step)
        step.extra_info["unknown_diagnostics"] = self._record_unknown_diagnostics_probe(step, commands, categories)

    def _action_taxonomy_summary(self) -> dict[str, Any]:
        sorted_counts = dict(sorted(self._action_counts.items(), key=lambda item: item[1], reverse=True))
        dominant_action = max(sorted_counts, key=sorted_counts.get) if sorted_counts else "none"
        command_total = max(self._action_total_commands, 1)
        exploration_count = (
            sorted_counts.get("search", 0) + sorted_counts.get("read_file", 0) + sorted_counts.get("navigate", 0)
        )
        execution_count = (
            sorted_counts.get("edit", 0) + sorted_counts.get("test", 0) + sorted_counts.get("python", 0)
        )
        return {
            "counts": sorted_counts,
            "steps_observed": self._action_step_count,
            "total_commands": self._action_total_commands,
            "dominant_action": dominant_action,
            "exploration_share": exploration_count / command_total,
            "execution_share": execution_count / command_total,
        }

    def _token_efficiency_summary(self) -> dict[str, Any]:
        step_count = len(self._token_snapshots)
        input_tokens = sum(item[0] for item in self._token_snapshots)
        output_tokens = sum(item[1] for item in self._token_snapshots)
        total_tokens = input_tokens + output_tokens
        totals_per_step = [a + b for a, b in self._token_snapshots]
        return {
            "total_input_tokens": input_tokens,
            "total_output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "steps_observed": step_count,
            "avg_tokens_per_step": (total_tokens / step_count) if step_count else 0.0,
            "median_tokens_per_step": self._median(totals_per_step),
            "zero_delta_steps": self._token_zero_delta_steps,
            "output_token_share": (output_tokens / total_tokens) if total_tokens else 0.0,
            "useful_token_share": (self._token_useful_total / total_tokens) if total_tokens else 0.0,
        }

    def _observation_bloat_summary(self) -> dict[str, Any]:
        steps = sum(self._obs_size_buckets.values())
        large_or_huge = self._obs_size_buckets.get("large", 0) + self._obs_size_buckets.get("huge", 0)
        return {
            "size_buckets": self._obs_size_buckets,
            "steps_observed": steps,
            "total_observation_chars": self._obs_total_chars,
            "avg_observation_chars": (self._obs_total_chars / steps) if steps else 0.0,
            "max_observation_chars": self._obs_max_chars,
            "large_or_huge_ratio": (large_or_huge / steps) if steps else 0.0,
        }

    def _iteration_waste_summary(self) -> dict[str, Any]:
        steps = max(len(self._token_snapshots), 1)
        total_waste_events = self._repeated_edits + self._reverts + self._duplicate_actions + self._duplicate_observations
        total_reread = self._reread_immediate + self._reread_after_edit + self._reread_distant
        total_redundancy_tokens = (
            self._post_submit_tokens
            + self._reread_tokens
            + self._repeated_git_check_tokens
            + self._duplicate_test_same_result_tokens
            + self._duplicate_python_verify_tokens
        )
        total_input = sum(item[0] for item in self._token_snapshots) if self._token_snapshots else 0
        return {
            "repeated_edits": self._repeated_edits,
            "reverts": self._reverts,
            "duplicate_actions": self._duplicate_actions,
            "duplicate_observations": self._duplicate_observations,
            "distinct_edited_targets": len(set(self._edit_file_history)),
            "total_waste_events": total_waste_events,
            "waste_event_rate": total_waste_events / steps,
            "redundancy_verification": {
                "post_submit": {
                    "steps": self._post_submit_steps,
                    "tokens": self._post_submit_tokens,
                },
                "file_reread": {
                    "immediate": self._reread_immediate,
                    "after_edit": self._reread_after_edit,
                    "distant": self._reread_distant,
                    "total": total_reread,
                    "tokens": self._reread_tokens,
                },
                "git_checks": {
                    "total_checks": self._git_check_count,
                    "repeated": self._repeated_git_checks,
                    "tokens": self._repeated_git_check_tokens,
                },
                "test_commands": {
                    "duplicate_same_result": self._duplicate_test_same_result,
                    "duplicate_same_result_tokens": self._duplicate_test_same_result_tokens,
                    "retest_different_result": self._retest_different_result,
                    "retest_different_result_tokens": self._retest_different_result_tokens,
                },
                "python_verify": {
                    "duplicate_same_result": self._duplicate_python_verify,
                    "tokens": self._duplicate_python_verify_tokens,
                },
                "total_redundancy_tokens": total_redundancy_tokens,
                "redundancy_token_ratio": (total_redundancy_tokens / total_input) if total_input else 0.0,
            },
        }

    def _context_pressure_summary(self) -> dict[str, Any]:
        if not self._query_sizes:
            return {
                "steps_observed": 0,
                "initial_query_chars": 0,
                "final_query_chars": 0,
                "peak_query_chars": 0,
                "avg_query_chars": 0.0,
                "total_growth_chars": 0,
                "avg_growth_per_step": 0.0,
                "positive_growth_steps": 0,
                "pressure_index": 0.0,
                "max_single_step_growth": 0,
            }
        first = self._query_sizes[0]
        last = self._query_sizes[-1]
        peak = max(self._query_sizes)
        deltas = self._query_size_deltas[1:] if len(self._query_size_deltas) > 1 else []
        return {
            "steps_observed": len(self._query_sizes),
            "initial_query_chars": first,
            "final_query_chars": last,
            "peak_query_chars": peak,
            "avg_query_chars": sum(self._query_sizes) / len(self._query_sizes),
            "total_growth_chars": last - first,
            "avg_growth_per_step": (last - first) / (len(self._query_sizes) - 1) if len(self._query_sizes) > 1 else 0.0,
            "positive_growth_steps": sum(1 for value in deltas if value > 0),
            "pressure_index": (last / first) if first else 0.0,
            "max_single_step_growth": max(deltas) if deltas else 0,
        }

    def _unknown_diagnostics_summary(self) -> dict[str, Any]:
        if self._unknown_steps == 0:
            return {
                "unknown_steps": 0,
                "reason_counts": {},
                "top_reason": "none",
                "error_signal_ratio": 0.0,
                "test_signal_ratio": 0.0,
                "long_observation_ratio": 0.0,
            }
        sorted_counts = dict(sorted(self._unknown_reason_counts.items(), key=lambda item: item[1], reverse=True))
        top_reason = next(iter(sorted_counts))
        return {
            "unknown_steps": self._unknown_steps,
            "reason_counts": sorted_counts,
            "top_reason": top_reason,
            "error_signal_ratio": self._unknown_error_signal_steps / self._unknown_steps,
            "test_signal_ratio": self._unknown_test_signal_steps / self._unknown_steps,
            "long_observation_ratio": self._unknown_long_observation_steps / self._unknown_steps,
        }

    def _probe_summary(self) -> dict[str, Any]:
        return {
            "action_taxonomy": self._action_taxonomy_summary(),
            "token_efficiency": self._token_efficiency_summary(),
            "observation_bloat": self._observation_bloat_summary(),
            "iteration_waste": self._iteration_waste_summary(),
            "context_pressure": self._context_pressure_summary(),
            "unknown_diagnostics": self._unknown_diagnostics_summary(),
        }

    def _thermo_probe_summary(self) -> dict[str, Any]:
        useful = self._thermo_stats.get("useful_work", 0)
        friction = self._thermo_stats.get("friction_loss", 0)
        unknown = self._thermo_stats.get("unknown", 0)
        total = useful + friction + unknown
        denom = useful + friction
        return {
            "useful_work": useful,
            "friction_loss": friction,
            "unknown": unknown,
            "total_classified_steps": total,
            "friction_ratio": (friction / total) if total else 0.0,
            "carnot_efficiency_proxy": (useful / denom) if denom else 0.0,
        }

    def handle_action(self, step: StepOutput) -> StepOutput:
        """Runs an action proposed by the agent in the environment and returns the corresponding output.

        Args:
            action: command to run in bash shell
            output: output from model (only used for error handling)

        Returns:
            action_execution_output: action execution output
        """
        if self.tools.should_block_action(step.action):
            raise _BlockedActionError()

        if step.action.strip() == "exit":
            self.logger.info("Exiting agent")
            step.done = True
            step.observation = "Exited"
            step.exit_status = "exit_command"
            assert self._env is not None
            step.state = self.tools.get_state(env=self._env)  # for history
            return step

        assert self._env is not None
        self._chook.on_action_started(step=step)
        execution_t0 = time.perf_counter()
        run_action: str = self.tools.guard_multiline_input(step.action).strip()
        try:
            step.observation = self._env.communicate(
                input=run_action,
                timeout=self.tools.config.execution_timeout,
                check="raise" if self._always_require_zero_exit_code else "ignore",
            )
        except CommandTimeoutError:
            self._n_consecutive_timeouts += 1
            if self._n_consecutive_timeouts >= self.tools.config.max_consecutive_execution_timeouts:
                msg = "Exiting agent due to too many consecutive execution timeouts"
                self.logger.critical(msg)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            try:
                self._env.interrupt_session()
            except Exception as f:
                self.logger.exception("Failed to interrupt session after command timeout: %s", f, exc_info=True)
                step.execution_time = time.perf_counter() - execution_t0
                self._total_execution_time += step.execution_time
                raise
            step.observation = Template(self.templates.command_cancelled_timeout_template).render(
                **self._get_format_dict(),
                timeout=self.tools.config.execution_timeout,
                command=run_action,
            )
        else:
            self._n_consecutive_timeouts = 0
        step.execution_time = time.perf_counter() - execution_t0
        self._total_execution_time += step.execution_time
        self._chook.on_action_executed(step=step)
        step.state = self.tools.get_state(env=self._env)

        if RETRY_WITH_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITH_OUTPUT_TOKEN, "")
            raise _RetryWithOutput()
        elif RETRY_WITHOUT_OUTPUT_TOKEN in step.observation:
            step.observation = step.observation.replace(RETRY_WITHOUT_OUTPUT_TOKEN, "")
            raise _RetryWithoutOutput()
        elif EXIT_FORFEIT_TOKEN in step.observation:
            raise _ExitForfeit()

        step = self.handle_submission(step)
        self._record_thermo_probe(step)
        self._record_multidim_probes(step)
        return step

    def forward(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model without handling errors.

        All exceptions raised will contain the `StepOutput` object
        with some of the attributes set.

        Args:
            history: history to query the model with

        Returns:
            step_output: step output
        """
        if self._total_execution_time > self.tools.config.total_execution_timeout:
            raise _TotalExecutionTimeExceeded()

        # we continuously add actions, output etc. to the step object
        # because some of the specific exception handling requires some of these
        # attributes (e.g., if we want to requery the model for a bash syntax error, we
        # need to have the previous model output to format the requery template)
        step = StepOutput()
        step.query = copy.deepcopy(history)
        try:
            # Forward model and get actions
            self._chook.on_model_query(messages=history, agent=self.name)
            # todo: Add all options to the extra info
            if self._action_sampler is not None:
                assert self._problem_statement is not None
                best = self._action_sampler.get_action(
                    problem_statement=self._problem_statement,
                    trajectory=self.trajectory,
                    history=history,
                )
                output = best.completion
                # todo: Handle history and trajectory
                step.extra_info.update(best.extra_info)
            else:
                output = self.model.query(history)  # type: ignore
            step.output = output["message"]
            # todo: Can't I override the parser in __init__?
            step.thought, step.action = self.tools.parse_actions(output)
            step.thinking_blocks = output.get("thinking_blocks", [])
            if output.get("tool_calls") is not None:
                step.tool_call_ids = [call["id"] for call in output["tool_calls"]]
                step.tool_calls = output["tool_calls"]
            self.logger.info(f"💭 THOUGHT\n{step.thought}\n\n🎬 ACTION\n{step.action.strip()}")
            self._chook.on_actions_generated(step=step)
            return self.handle_action(step)
        except Exception as e:
            if step.action == step.thought == "":
                # Probably the parsing failed/no action included. Let's still fill in thought
                # so that trajectory viewers have something to show us for this step.
                step.thought = step.output
            # Attach the step object to the exception
            e.step = step  # type: ignore
            raise

    def forward_with_handling(self, history: list[dict[str, str]]) -> StepOutput:
        """Forward the model and handle errors, requerying the model if we can.
        For example, if the model outputs a bash command that has syntax errors,
        we will not execute it but requery the model for a corrected command.

        Note: This will update the trajectory, but not the history.

        Args:
            history: history to forward

        Returns:
            step_output: step output
        """

        def handle_error_with_autosubmission(
            exit_status: str, message: str, exception: Exception | None = None
        ) -> StepOutput:
            """Attempts to autosubmit (extract patch from the environment) and stops the loop."""
            self.logger.warning(message)
            step: StepOutput = getattr(exception, "step", StepOutput())
            if not step.thought:
                step.thought = message
            if not step.output:
                step.output = message
            step.exit_status = exit_status
            step.done = True
            self._record_multidim_probes(step)
            return self.attempt_autosubmission_after_error(step)

        def handle_error_with_retry(exception: Exception, template: str, n_requeries: int) -> list[dict[str, str]]:
            """Requeries the model if the error is a format/blocklist/bash syntax error."""
            self.logger.warning("Requerying model after %s (%dth requery)", type(exception).__name__, n_requeries)
            step: StepOutput = getattr(exception, "step", StepOutput())
            self._record_multidim_probes(step)
            self.add_step_to_trajectory(step)
            exception_message = getattr(exception, "message", "")
            if not exception_message:
                try:
                    exception_message = exception.args[0]
                except (IndexError, AttributeError):
                    pass
            return self.get_model_requery_history(
                error_template=template,
                **step.to_template_format_dict(),
                **getattr(exception, "extra_info", {}),
                exception_message=exception_message,
            )

        n_format_fails = 0
        while n_format_fails < self.max_requeries:
            try:
                return self.forward(history)

            # Errors that are raised

            except KeyboardInterrupt:
                raise
            except EOFError:
                raise

            # Errors that cause requery

            except FormatError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.format_error_template, n_requeries=n_format_fails
                )
            except _BlockedActionError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e, template=self.tools.config.filter.blocklist_error_template, n_requeries=n_format_fails
                )
            except ContentPolicyViolationError:
                self.logger.warning("Content policy violation, trying to resample")
                n_format_fails += 1
                # Try if simply resampling helps here
                pass
            except BashIncorrectSyntaxError as e:
                n_format_fails += 1
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.shell_check_error_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithOutput as e:
                history = handle_error_with_retry(
                    exception=e,
                    template=self.templates.next_step_template,
                    n_requeries=n_format_fails,
                )
            except _RetryWithoutOutput:
                pass
                # Requery with the same template as the last step

            # Errors that cause exit

            except _ExitForfeit as e:
                self.logger.info("Exiting due to forfeit")
                return handle_error_with_autosubmission(
                    "exit_forfeit",
                    "Exiting due to forfeit",
                    exception=e,
                )

            except _TotalExecutionTimeExceeded as e:
                self.logger.exception("Exiting due to total execution time exceeded", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_total_execution_time",
                    "Exit due to total execution time exceeded",
                    exception=e,
                )

            except CommandTimeoutError as e:
                self.logger.exception("Exiting due to multiple consecutive command timeouts", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_command_timeout",
                    "Exit due to multiple consecutive command timeouts",
                    exception=e,
                )

            except ContextWindowExceededError as e:
                return handle_error_with_autosubmission(
                    "exit_context",
                    "Exit due to context window",
                    exception=e,
                )
            except TotalCostLimitExceededError:
                raise
            except BudgetExhaustedError as e:
                return handle_error_with_autosubmission(
                    "exit_token_budget",
                    "Exit due to token budget exhaustion",
                    exception=e,
                )
            except CostLimitExceededError as e:
                return handle_error_with_autosubmission(
                    "exit_cost",
                    "Exit due to cost limit",
                    exception=e,
                )
            except RetryError as e:
                self.logger.exception(f"Exiting due to retry error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_api",
                    f"Exit due to retry error: {e}",
                    exception=e,
                )
            except SwerexException as e:
                self.logger.exception(f"Exiting due to environment error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_environment_error",
                    f"Exit due to environment error: {e}",
                    exception=e,
                )
            except RuntimeError as e:
                self.logger.exception(f"Exiting due to runtime error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to runtime error: {e}",
                    exception=e,
                )
            except Exception as e:
                self.logger.exception(f"Exiting due to unknown error: {e}", exc_info=True)
                return handle_error_with_autosubmission(
                    "exit_error",
                    f"Exit due to unknown error: {e}",
                    exception=e,
                )
        self.logger.exception(
            "Exit due to repeated format/blocklist/bash syntax errors",
            exc_info=True,
        )
        return handle_error_with_autosubmission(
            "exit_format",
            "Exit due to repeated format/blocklist/bash syntax errors",
        )

    def add_step_to_trajectory(self, step: StepOutput) -> None:
        trajectory_step = TrajectoryStep(
            {
                "action": step.action,
                "observation": step.observation,
                "response": step.output,
                "thought": step.thought,
                "execution_time": step.execution_time,
                "state": step.state,
                "query": step.query,
                "extra_info": step.extra_info,
            },
        )
        self.trajectory.append(trajectory_step)

    def step(self) -> StepOutput:
        """Run a step of the agent. This is a wrapper around `self.forward_with_handling`
        with additional bookkeeping:

        1. Update message history with performed action and observation
        2. Update trajectory with the final executed result
        3. Update the info dictionary

        Returns:
            step_output: step output (same as the output of `self.forward_with_handling`)
        """

        assert self._env is not None
        self._chook.on_step_start()

        n_step = len(self.trajectory) + 1
        self.logger.info("=" * 25 + f" STEP {n_step} " + "=" * 25)
        step_output = self.forward_with_handling(self.messages)
        self.add_step_to_history(step_output)

        self.info["submission"] = step_output.submission
        self.info["exit_status"] = step_output.exit_status  # type: ignore
        self.info.update(self._get_edited_files_with_context(patch=step_output.submission or ""))  # type: ignore
        model_stats = self.model.stats.model_dump()
        model_stats["total_tokens"] = model_stats.get("tokens_sent", 0) + model_stats.get("tokens_received", 0)
        self.info["model_stats"] = model_stats
        self.info["thermo_probe"] = self._thermo_probe_summary()
        self.info["probe_summary"] = self._probe_summary()

        self.add_step_to_trajectory(step_output)

        self._chook.on_step_done(step=step_output, info=self.info)
        return step_output

    def run(
        self,
        env: SWEEnv,
        problem_statement: ProblemStatement | ProblemStatementConfig,
        output_dir: Path = Path("."),
    ) -> AgentRunResult:
        """Run the agent on a problem instance. This method contains the
        main loop that repeatedly calls `self._step` until the problem is solved.

        Args:
            setup_args: Arguments to pass to the agent's setup method.
            env: The environment to run the agent on.
            traj_dir: Directory to save the trajectory to
        """
        self.setup(env=env, problem_statement=problem_statement, output_dir=output_dir)

        # Run action/observation loop
        self._chook.on_run_start()
        step_output = StepOutput()
        while not step_output.done:
            step_output = self.step()
            self.save_trajectory()
        self._chook.on_run_done(trajectory=self.trajectory, info=self.info)

        self.logger.info("Trajectory saved to %s", self.traj_path)

        # Here we want to return the "global" information (e.g., submission should
        # be the best submission instead of the last one, etc.), so we get it from the traj file
        data = self.get_trajectory_data()
        return AgentRunResult(info=data["info"], trajectory=data["trajectory"])
