#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""LLM model benchmarking."""

import inspect
import logging
import operator
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

import pynguin.ga.testcasechromosome as tcc
import pynguin.generator
from pynguin import configuration as config
from pynguin.ga.algorithms.llmosalgorithm import LLMOSAAlgorithm
from pynguin.large_language_model.llmagent import LLMAgent, get_module_path, get_module_source_code
from pynguin.large_language_model.prompts.uncoveredtargetsprompt import UncoveredTargetsPrompt
from pynguin.slicer.statementslicingobserver import RemoteStatementSlicingObserver
from pynguin.utils.api_key_resolver import get_llm_url
from pynguin.utils.generic.genericaccessibleobject import GenericCallableAccessibleObject
from pynguin.utils.report import CoverageReport, LineAnnotation, get_coverage_report

if TYPE_CHECKING:
    from pynguin.ga.algorithms.generationalgorithm import GenerationAlgorithm

_LOGGER = logging.getLogger(__name__)


class PromptBenchmarkExperiment(BenchmarkExperiment):
    """Benchmarks the performance of a prompt."""

    def __init__(
        self,
        prompt: type[UncoveredTargetsPrompt],
    ) -> None:
        """Benchmark the performance of a prompt.

        Args:
            prompt: The prompt function to benchmark.
        """
        self.prompt = prompt
        self.provider = config.LLMProvider.OLLAMA
        self.model = "qwen2.5-coder:1.5b"
        self.only_llm = False

    def setup(self):  # noqa: D102
        if self.provider == config.LLMProvider.OLLAMA and not any(
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
        _LOGGER.debug("Running prompt experiment (%s)", self.prompt.__name__)
        _LOGGER.debug("Sample: %s", sample)
        try:
            test_cluster = sample.test_cluster

            config.configuration.large_language_model.provider = self.provider
            config.configuration.large_language_model.model_name = self.model
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

            algorithm.model.clear_cache()

            llm_chromosomes = self._target_uncovered_callables(algorithm)
            if not llm_chromosomes:
                _LOGGER.warning("No LLM chromosomes generated")
            algorithm._population += llm_chromosomes  # noqa: SLF001
            algorithm._archive.update(algorithm._population)  # noqa: SLF001

            coverage_after = algorithm.create_test_suite(
                algorithm._archive.solutions  # noqa: SLF001
            ).get_coverage()

            _LOGGER.info(
                "Prompt experiment completed (%s), coverage=%s",
                self.prompt.__name__,
                coverage_after,
            )
            return BenchmarkExperimentResult(success=True, score=coverage_after)

        except Exception:
            _LOGGER.exception("Error running prompt experiment")
            return BenchmarkExperimentResult(success=False)

    def _target_uncovered_callables(
        self, algorithm: LLMOSAAlgorithm
    ) -> list[tcc.TestCaseChromosome]:
        """Identifies uncovered targets, queries an LLM for test cases.

         and processes the results into a list of test case chromosomes.

        Returns:
            A list of `TestCaseChromosome` objects derived from the LLM query results.
        """
        solutions_test_suite = algorithm.create_test_suite(algorithm._archive.solutions)  # noqa: SLF001

        def coverage_in_range(start_line: int, end_line: int) -> tuple[int, int]:
            """Calculate the total and covered coverage points for a given line range.

            Args:
                start_line: The first line in the range, inclusive.
                end_line: The last line in the range, inclusive.

            Returns:
                A tuple of (covered points, total points).
            """
            total_coverage_points = 0
            covered_coverage_points = 0
            for line_annot in line_annotations:
                if start_line <= line_annot.line_no <= end_line:
                    total_coverage_points += line_annot.total.existing
                    covered_coverage_points += line_annot.total.covered
            return covered_coverage_points, total_coverage_points

        def calculate_gao_coverage_map() -> dict[GenericCallableAccessibleObject, float]:
            """Calculate the coverage ratio for each GenericCallableAccessibleObject.

            Returns:
                A dictionary mapping accessible objects to their coverage ratios.
            """
            gao_coverage = {}
            for gao in algorithm.test_cluster.accessible_objects_under_test:
                if isinstance(gao, GenericCallableAccessibleObject):
                    try:
                        source_lines, start_line = inspect.getsourcelines(gao.callable)
                        end_line = start_line + len(source_lines) - 1
                        covered, total = coverage_in_range(start_line, end_line)
                        coverage_ratio = covered / total if total > 0 else 0
                    except (TypeError, OSError):
                        coverage_ratio = 0
                    gao_coverage[gao] = coverage_ratio
            return gao_coverage

        def filter_gao_by_coverage(
            gao_coverage: dict[GenericCallableAccessibleObject, float],
        ) -> dict[GenericCallableAccessibleObject, float]:
            """Filter GenericCallableAccessibleObjects by their coverage ratio.

            Args:
                gao_coverage: A dictionary of objects and their coverage ratios.

            Returns:
                A filtered dictionary of objects with coverage below the threshold.
            """
            return {
                gao: coverage
                for gao, coverage in sorted(gao_coverage.items(), key=operator.itemgetter(1))
                if coverage < config.configuration.large_language_model.coverage_threshold
            }

        # Main logic
        coverage_report: CoverageReport = get_coverage_report(
            solutions_test_suite,
            algorithm.executor.subject_properties,
            set(config.configuration.statistics_output.coverage_metrics),
        )
        line_annotations: list[LineAnnotation] = coverage_report.line_annotations

        gao_coverage_map = calculate_gao_coverage_map()
        filtered_gao_coverage_map = filter_gao_by_coverage(gao_coverage_map)

        llm_query_results = self._call_llm_for_uncovered_targets(filtered_gao_coverage_map)

        return algorithm.model.llm_test_case_handler.get_test_case_chromosomes_from_llm_results(
            llm_query_results=llm_query_results,
            test_cluster=algorithm.test_cluster,
            test_factory=algorithm._test_factory,  # noqa: SLF001
            fitness_functions=algorithm._test_case_fitness_functions,  # noqa: SLF001
            coverage_functions=algorithm._test_suite_coverage_functions,  # noqa: SLF001
        )

    def _call_llm_for_uncovered_targets(
        self, gao_coverage_map: dict[GenericCallableAccessibleObject, float]
    ):
        """Queries the language model for uncovered targets.

        Args:
            gao_coverage_map (dict): Maps callable objects to coverage percentages.

        Returns:
            Any: Result of the query based on the constructed prompt.
        """
        module_code = get_module_source_code()
        module_path = get_module_path()
        prompt = self.prompt(list(gao_coverage_map.keys()), module_code, str(module_path))
        return LLMAgent(provider=self.provider, model_name=self.model).query(prompt)

    def __str__(self) -> str:
        return f"Prompt experiment ({self.prompt.__name__})"
