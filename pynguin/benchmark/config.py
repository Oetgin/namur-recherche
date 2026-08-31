#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Benchmark configurations."""

# TODO (Oetgin): Refactor, config should not be used to run the benchmark, but only to configure it.

import logging
import sys
from enum import Enum
from pathlib import Path
from time import localtime, strftime
from typing import ClassVar

import simple_parsing
from typing_extensions import override

from pynguin.large_language_model.prompts.uncoveredtargetsprompt import UncoveredTargetsPrompt
from pynguin.utils.generic.genericaccessibleobject import (
    GenericCallableAccessibleObject,
)

if __package__ in {None, ""}:
    _PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PACKAGE_ROOT))
    sys.path.insert(0, str(_PACKAGE_ROOT / "src"))

from rich.logging import RichHandler

from benchmark.benchmark import BenchmarkSuite, Dataset
from benchmark.benchmarks.model import ModelBenchmarkExperiment
from benchmark.benchmarks.prompt import PromptBenchmarkExperiment
from benchmark.export import CSV
from pynguin.configuration import LLMProvider

_LOGGER = logging.getLogger(__name__)


DATASET_RELATIVE_PATHS = [
    # "scikit-learn/sklearn/preprocessing/_data.py",  # scikit-learn requires a building step
    # "scikit-learn/sklearn/utils/validation.py",  # scikit-learn requires a building step
    # "scikit-learn/sklearn/pipeline.py",  # scikit-learn requires a building step
    # "tensorflow/tensorflow/python/util/nest.py",  # tensorflow requires a building step
    # "tensorflow/tensorflow/python/util/dispatch.py",  # tensorflow requires a building step
    # "tensorflow/tensorflow/python/util/variable_utils.py",  # tensorflow requires a building step
    # "tensorflow/tensorflow/python/util/object_identity.py",  # tensorflow requires a building step
    # "vllm/vllm/v1/structured_output/request.py",  # circular import error on WSL  # noqa: ERA001
    # "vllm/vllm/v1/spec_decode/utils.py",  # circular import error on WSL  # noqa: ERA001
    "pytorch/torch/_numpy/_ndarray.py",
    "pytorch/torch/_numpy/_getlimits.py",
    "pytorch/torch/nn/attention/bias.py",
    "pytorch/torch/nn/attention/flex_attention.py",
    "Tabular-data-generation/src/_ForestDiffusion/diffusion_with_trees_class.py",
    "Tabular-data-generation/src/tabgan/adversarial_model.py",
    "Tabular-data-generation/src/_ctgan/transformer.py",
    "transformers/src/transformers/tokenization_utils_base.py",
    "transformers/src/transformers/pipelines/text_generation.py",
    "transformers/src/transformers/data/metrics/squad_metrics.py",
    "pytorch/torch/nn/modules/rnn.py",
    "pytorch/torch/_numpy/_util.py",
    "pytorch/torch/_numpy/_funcs.py",
    "pytorch/torch/_numpy/_ufuncs.py",
    "pytorch/torch/_numpy/_normalizations.py",
]

_DATASET_ROOT = Path(__file__).resolve().parent / "samples" / "github"

DATASET = Dataset(*(_DATASET_ROOT / relative_path for relative_path in DATASET_RELATIVE_PATHS))


class ModelBenchmark:
    """Configuration for benchmarking different LLMs."""

    _MODELS: ClassVar[dict] = {
        # Qwen
        "qwen2_5_coder__0_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:0.5b"),
        "qwen2_5_coder__1_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:1.5b"),
        "qwen2_5_coder__3b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:3b"),
        "qwen2_5_coder__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:7b"),
        "qwen2_5__0_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5:0.5b"),
        "qwen2_5__1_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5:1.5b"),
        "qwen2_5__3b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5:3b"),
        "qwen2_5__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5:7b"),
        "qwen3__0_6b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3:0.6b"),
        "qwen3__1_7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3:1.7b"),
        "qwen3__4b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3:4b"),
        "qwen3_5__0_8b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3.5:0.8b"),
        "qwen3_5__2b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3.5:2b"),
        "qwen3_5__4b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3.5:4b"),
        # Codellama / Codegemma
        "codellama__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "codellama:7b"),
        "codegemma__2b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "codegemma:2b"),
        "codegemma__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "codegemma:7b"),
        # Gemma
        "gemma4__e2b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "gemma4:e2b"),
        "gemma4__e4b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "gemma4:e4b"),
        "gemma4__12b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "gemma4:12b"),
        "gemma4__26b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "gemma4:26b"),
        "gemma4__31b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "gemma4:31b"),
        # Olmo
        "olmo_3__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "olmo-3:7b"),
        # LFM
        "lfm2_5__8b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "lfm2.5:8b"),
        # Phi
        "phi4_mini__3_8b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "phi4-mini:3.8b"),
        "phi4_mini_reasoning__3_8b": ModelBenchmarkExperiment(
            LLMProvider.OLLAMA, "phi4-mini-reasoning:3.8b"
        ),
        # Deepseek
        "deepseek_coder__1_3b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "deepseek-coder:1.3b"),
        "deepseek_coder__6_7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "deepseek-coder:6.7b"),
        "deepseek_r1__1_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "deepseek-r1:1.5b"),
        "deepseek_r1__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "deepseek-r1:7b"),
    }

    _TO_RUN: ClassVar[set[str]] = {
        "qwen2_5_coder__7b",
        "qwen2_5__3b",
        "qwen2_5__1_5b",
        "qwen3__4b",
        "qwen3_5__4b",
        "gemma4__12b",
        "gemma4__26b",
        "gemma4__31b",
    }

    experiments: ClassVar[list] = []
    for name, experiment in _MODELS.items():
        if name in _TO_RUN:
            experiments.append(experiment)

    benchmark = BenchmarkSuite(experiments, DATASET, n_runs=30, max_samples=10)


class PromptBenchmark:
    """Configuration for benchmarking different prompts."""

    class DefaultPrompt(UncoveredTargetsPrompt):
        """Default prompt implementation."""

        def __init__(
            self,
            callables: list[GenericCallableAccessibleObject],
            module_code: str,
            module_path: str,
        ):
            """Initializes the prompt.

            Args:
                callables (list[GenericCallableAccessibleObject]): List of
                    uncovered callables.
                module_path (str): Path to the module.
                module_code (str): Source code of the module.
            """
            super().__init__(callables, module_code, module_path)

        @override
        def build_prompt(self) -> str:
            """Builds the prompt message."""
            callables_list = self.build_callables_prompt_section()
            callables_section = "\n".join(callables_list)

            return (
                f"Write unit tests for the following callables that "
                f" Pynguin failed to cover:\n"
                f"{callables_section}\n"
                f"Module path: `{self.module_path}`\n"
                f"Module source code: `{self.module_code}`"
            )

    class NoCompressionPrompt(UncoveredTargetsPrompt):
        """Prompt implementation without compression."""

        def __init__(
            self,
            callables: list[GenericCallableAccessibleObject],
            module_code: str,
            module_path: str,
        ):
            """Initializes the prompt.

            Args:
                callables (list[GenericCallableAccessibleObject]): List of
                    uncovered callables.
                module_path (str): Path to the module.
                module_code (str): Source code of the module.
            """
            super().__init__(callables, module_code, module_path)

        @override
        def build_prompt(self) -> str:
            """Builds the prompt message."""
            callables_list = self.build_callables_prompt_section()
            callables_section = "\n".join(callables_list)

            # TODO (Oetgin): Finalize and test prompt compression

            return f"""
You are writing tests to be used as seed for a SBST algorithm. Your goal is to improve as much as possible the coverage of the tests.
Write unit tests for the following callables that Pynguin failed to cover:
{callables_section}
Module path: `{self.module_path}`
Module source code:
```python
{self.module_code}
```
You answer will be parsed for mutations, so here are the guidelines you need to follow:
- Answer in a code block only using; one function for each test case, with NO ARGUMENTS, NO HELPER FUNCTIONS AND NO CLASSES.
- If needed, instantiate vars in the body of the test func or use pytest.parametrize, but DO NOT USE ANY OTHER PYTEST FEATURE (e.g. DO NOT USE FIXTURES), as that will make the parsing fail.
- Do not rewrite the SUT's code in the tests. If you want for example to call a function or instanciante a class, import it.
- Without explaining, answer in simple, concise assertion tests, split in small functions. Follow the Arrange, Act, Assert pattern.

Here are some examples; *NEVER DO*:
def func_all_tests(param):
    var0 = param
    ...

*INSTEAD DO*:
I need to test the function `func` that takes a parameter and returns a value. I will test the following scenarios: ...
```python
from module_to_test import func
def test_feat1():
    var0 = ...
    var1 = func(var0)
    assert ...
```

REMEMBER: NO ARGUMENTS IN YOUR TEST FUNCTIONS (if you don't use pytest.parametrize)"
"""  # noqa: E501

    _PROMPTS: ClassVar[dict] = {
        "default": PromptBenchmarkExperiment(DefaultPrompt),
        "ours": PromptBenchmarkExperiment(UncoveredTargetsPrompt),
        "no_compression": PromptBenchmarkExperiment(NoCompressionPrompt),
    }

    _TO_RUN: ClassVar[set[str]] = {"default", "ours", "no_compression"}

    experiments: ClassVar[list] = []
    for name, experiment in _PROMPTS.items():
        if name in _TO_RUN:
            experiments.append(experiment)

    benchmark = BenchmarkSuite(experiments, DATASET, n_runs=30, max_samples=15)


class ClassicBenchmark:
    """Configuration for benchmarking classic Pynguin.

    Runs a model experiment with a single model.
    """

    benchmark = BenchmarkSuite(
        [ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:0.5b", only_llm=False)],
        DATASET,
        n_runs=30,
        max_samples=15,
    )


def _create_argument_parser() -> simple_parsing.ArgumentParser:
    parser = simple_parsing.ArgumentParser(
        add_option_string_dash_variants=simple_parsing.DashVariant.UNDERSCORE_AND_DASH,
        description="Benchmarking Pynguin.",
    )
    parser.add_argument(
        "--to-run",
        nargs="+",
        choices=["model", "prompt", "classic"],
        default=["prompt", "classic"],
        help="Which benchmarks to run. Default: ['prompt', 'classic']",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Main function to run configured benchmarks.

    Args:
        argv: List of command-line arguments. If None, sys.argv is used.
    """
    if argv is None:
        argv = sys.argv
    if not argv[0].startswith("-"):
        argv = argv[1:]

    argument_parser = _create_argument_parser()
    parsed = argument_parser.parse_args(argv)

    class ExperimentType(Enum):
        MODEL = "model"
        PROMPT = "prompt"
        CLASSIC = "classic"

    to_run = {ExperimentType(arg) for arg in parsed.to_run}

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_dir.joinpath(".logs").mkdir(parents=True, exist_ok=True)

    class BenchmarkWhitelistFilter(logging.Filter):
        """Logging filter for benchmarking."""

        def filter(self, record: logging.LogRecord) -> bool:
            """Filters benchmarking records for logging purposes.

            Args:
                record (logging.LogRecord): Record to filter.

            Returns:
                bool: True if record is from the benchmarking logic.
            """
            return record.name == "__main__" or record.name.startswith("benchmark")

    log_formatter = logging.Formatter("%(asctime)s [%(threadName)s] [%(levelname)s] %(message)s")

    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(
        f"{out_dir}/.logs/{strftime('%d-%m-%Y_%H-%M-%S', localtime())}.log"
    )
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    rich_handler = RichHandler()
    rich_handler.setFormatter(log_formatter)
    rich_handler.addFilter(BenchmarkWhitelistFilter())
    root_logger.addHandler(rich_handler)

    logging.getLogger("pynguin").setLevel(logging.INFO)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if ExperimentType.CLASSIC in to_run:
        _LOGGER.info("Running classic benchmark")
        ClassicBenchmark.benchmark.run()
        _LOGGER.debug("Benchmark results: %s", ClassicBenchmark.benchmark.results)
        CSV.export(ClassicBenchmark.benchmark.results, out_dir / "classic_results.csv")

    if ExperimentType.PROMPT in to_run:
        _LOGGER.info("Running prompt benchmark")
        PromptBenchmark.benchmark.run()
        _LOGGER.debug("Benchmark results: %s", PromptBenchmark.benchmark.results)
        CSV.export(PromptBenchmark.benchmark.results, out_dir / "prompt_results.csv")

    if ExperimentType.MODEL in to_run:
        _LOGGER.info("Running model benchmark")
        ModelBenchmark.benchmark.run()
        _LOGGER.debug("Benchmark results: %s", ModelBenchmark.benchmark.results)
        CSV.export(ModelBenchmark.benchmark.results, out_dir / "model_results.csv")


if __name__ == "__main__":
    main(sys.argv)
