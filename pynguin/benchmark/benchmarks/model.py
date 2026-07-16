#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""LLM model benchmarking."""

import logging
import textwrap
from typing import TYPE_CHECKING

import ollama
from benchmark.benchmark import (
    BenchmarkExperiment,
    BenchmarkExperimentResult,
    Sample,
    measure_exec_time,
)
from rich.progress import (
    BarColumn,
    DownloadColumn,
    Progress,
    TextColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)

import pynguin.generator
from pynguin import configuration as config
from pynguin.configuration import LLMProvider
from pynguin.ga.algorithms.llmosalgorithm import LLMOSAAlgorithm
from pynguin.slicer.statementslicingobserver import RemoteStatementSlicingObserver
from pynguin.utils.api_key_resolver import get_llm_url

if TYPE_CHECKING:
    from pynguin.ga.algorithms.generationalgorithm import GenerationAlgorithm

_LOGGER = logging.getLogger(__name__)


class ModelBenchmarkExperiment(BenchmarkExperiment):
    """Benchmarks an AI model."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        *,
        temperature: float | None = None,
        only_llm: bool = True,
    ) -> None:
        """Benchmark different AI models.

        Args:
            provider (LLMProvider): Model provider.
            model (str): Model name.
            temperature (float | None, optional): Optional temperature parameter.
            only_llm (bool, optional): Only use LLM generated testcases.
                If False, will also use the standard algorithm and only call the LLM on plateaus.
                Defaults to True.
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.only_llm = only_llm

    def setup(self):  # noqa: D102
        _LOGGER.debug(
            "Setting up model experiment (model: %s, llm_url: %s)", self.model, get_llm_url()
        )
        if self.provider == LLMProvider.OLLAMA and not any(
            model.model == self.model for model in ollama.Client(host=get_llm_url()).list().models
        ):
            _LOGGER.info("Model %s not found. Pulling from Ollama...", self.model)
            progress_response = ollama.Client(host=get_llm_url()).pull(self.model, stream=True)

            pbar = Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(bar_width=None),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                DownloadColumn(),
                "•",
                TransferSpeedColumn(),
                "•",
                TimeRemainingColumn(),
            )
            task = pbar.add_task(f"Pulling model {self.model}")
            with pbar:
                for progress in progress_response:
                    if progress.completed is None:
                        pbar.update(task, description=progress.status)
                    else:
                        pbar.update(
                            task,
                            description=progress.status,
                            total=progress.total,
                            completed=progress.completed,
                        )

    @measure_exec_time
    def run(self, sample: Sample) -> BenchmarkExperimentResult:  # noqa: D102
        _LOGGER.debug("Running model experiment (model : %s)", self.model)
        _LOGGER.debug("Sample: %s", sample)
        try:
            test_cluster = sample.test_cluster

            config.configuration.large_language_model.provider = self.provider
            config.configuration.large_language_model.model_name = self.model
            if self.temperature is not None:
                config.configuration.large_language_model.temperature = self.temperature

            config.configuration.algorithm = config.Algorithm.LLMOSA
            config.configuration.module_name = sample.module_name

            if (setup_result := pynguin.generator._setup_and_check()) is None:  # noqa: SLF001
                _LOGGER.error("Setup failed")
                return BenchmarkExperimentResult(success=False)
            executor, test_cluster, constant_provider = setup_result
            coverage_metrics = config.configuration.statistics_output.coverage_metrics
            if config.CoverageMetric.CHECKED in coverage_metrics:
                executor.add_remote_observer(RemoteStatementSlicingObserver())

            algorithm: GenerationAlgorithm = (
                pynguin.generator._instantiate_test_generation_strategy(  # noqa: SLF001
                    executor, test_cluster, constant_provider
                )
            )

            if not isinstance(algorithm, LLMOSAAlgorithm):
                _LOGGER.error("Algorithm type is not LLMOSA")
                return BenchmarkExperimentResult(success=False)

            if self.only_llm:
                algorithm.model.clear_cache()

                llm_chromosomes = algorithm.target_uncovered_callables()
                if not llm_chromosomes:
                    _LOGGER.warning("No LLM chromosomes generated")
                algorithm._population += llm_chromosomes  # noqa: SLF001
                algorithm._archive.update(algorithm._population)  # noqa: SLF001

                coverage_after = algorithm.create_test_suite(
                    algorithm._archive.solutions  # noqa: SLF001
                ).get_coverage()

                _LOGGER.info(
                    "Model %s experiment completed, coverage=%s", self.model, coverage_after
                )
                return BenchmarkExperimentResult(success=True, score=coverage_after)

            algorithm.generate_tests()

            coverage_after = algorithm.create_test_suite(
                algorithm._archive.solutions  # noqa: SLF001
            ).get_coverage()

            return BenchmarkExperimentResult(success=True, score=coverage_after)

        except Exception:
            _LOGGER.exception("Error running model experiment (model : %s)", self.model)
            return BenchmarkExperimentResult(success=False)

    def __str__(self) -> str:
        return f"Model experiment ({self.model})"

    def __repr__(self) -> str:
        return textwrap.dedent(f"""ModelBenchmarkExperiment(
            provider={self.provider!r},
            model={self.model!r},
            temperature={self.temperature!r},
            only_llm={self.only_llm!r}
        )""")
