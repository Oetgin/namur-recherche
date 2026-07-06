import logging  # noqa: D100
import sys
from pathlib import Path
from time import localtime, strftime
from typing import ClassVar

if __package__ in {None, ""}:
    _PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(_PACKAGE_ROOT))
    sys.path.insert(0, str(_PACKAGE_ROOT / "src"))

from rich.logging import RichHandler

from benchmark.benchmark import BenchmarkSuite, Dataset
from benchmark.export import CSV
from benchmark.model.model import ModelBenchmarkExperiment
from pynguin.configuration import LLMProvider

_LOGGER = logging.getLogger(__name__)


class ModelBenchmark:
    """Configuration for benchmarking different LLMs."""

    _MODELS: ClassVar[dict] = {
        # Qwen
        "qwen2_5_coder__0_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:0.5b"),
        "qwen2_5_coder__1_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:1.5b"),
        # "qwen2_5_coder__3b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:3b"),
        # "qwen2_5_coder__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5-coder:7b"),
        # "qwen2_5__0_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5:0.5b"),
        # "qwen2_5__1_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5:1.5b"),
        # "qwen2_5__3b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5:3b"),
        # "qwen2_5__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen2.5:7b"),
        # "qwen3__0_6b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3:0.6b"),
        # "qwen3__1_7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3:1.7b"),
        # "qwen3__4b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3:4b"),
        # "qwen3_5__0_8b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3.5:0.8b"),
        # "qwen3_5__2b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3.5:2b"),
        # "qwen3_5__4b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "qwen3.5:4b"),
        # # Codellama / Codegemma
        # "codellama__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "codellama:7b"),
        # "codegemma__2b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "codegemma:2b"),
        # "codegemma__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "codegemma:7b"),
        # # Gemma
        # "gemma4__e2b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "gemma4:e2b"),
        # "gemma4__e4b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "gemma4:e4b"),
        # # Olmo
        # "olmo_3__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "olmo-3:7b"),
        # # LFM
        # "lfm2_5__8b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "lfm2.5:8b"),
        # # Phi
        # "phi4_mini__3_8b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "phi4-mini:3.8b"),
        # "phi4_mini_reasoning__3_8b": ModelBenchmarkExperiment(
        #     LLMProvider.OLLAMA, "phi4-mini-reasoning:3.8b"
        # ),
        # # Deepseek
        # "deepseek_coder__1_3b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "deepseek-coder:1.3b"),
        # "deepseek_coder__6_7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "deepseek-coder:6.7b"),
        # "deepseek_r1__1_5b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "deepseek-r1:1.5b"),
        # "deepseek_r1__7b": ModelBenchmarkExperiment(LLMProvider.OLLAMA, "deepseek-r1:7b"),
    }

    experiments: ClassVar[list] = list(_MODELS.values())

    dataset = Dataset(Path(__file__).resolve().parent / "model" / "samples")

    benchmark = BenchmarkSuite(experiments, dataset, n_runs=30)


if __name__ == "__main__":

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

    file_handler = logging.FileHandler(f".logs/{strftime('%d-%m-%Y_%H-%M-%S', localtime())}.log")
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    rich_handler = RichHandler()
    rich_handler.setFormatter(log_formatter)
    rich_handler.addFilter(BenchmarkWhitelistFilter())
    root_logger.addHandler(rich_handler)

    logging.getLogger("pynguin").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _LOGGER.info("Running model benchmark")
    ModelBenchmark.benchmark.run()
    CSV.export(ModelBenchmark.benchmark.results)
