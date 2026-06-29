#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2026 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""LLM model benchmarking."""

import logging
from pathlib import Path

import ollama
from benchmark.benchmark import (
    BenchmarkExperiment,
    BenchmarkExperimentResult,
    Dataset,
    Sample,
    measure_exec_time,
)

from pynguin.configuration import LLMProvider
from pynguin.large_language_model.llmagent import LLMAgent
from pynguin.large_language_model.llmtestcasehandler import LLMTestCaseHandler
from pynguin.large_language_model.parsing.deserializer import deserialize_code_to_testcases
from pynguin.large_language_model.prompts.testcasegenerationprompt import TestCaseGenerationPrompt

_LOGGER = logging.getLogger(__name__)


class ModelBenchmarkExperiment(BenchmarkExperiment):
    """Benchmarks an AI model."""

    def __init__(
        self,
        provider: LLMProvider,
        model: str,
        temperature: float | None = None,
    ) -> None:
        """Benchark different AI models.

        Args:
            provider (LLMProvider): Model provider.
            model (str): Model name.
            temperature (float | None, optional): Optional temperature parameter.
        """
        self.provider = provider
        self.model = model
        self.temperature = temperature

    def setup(self):
        if self.provider == LLMProvider.OLLAMA and any(
            model == self.model for model in ollama.Client().list().models
        ):
            _LOGGER.info(f"Model {self.model} not found. Pulling from Ollama...")
            ollama.Client().pull(self.model)

    @measure_exec_time
    def run(self, sample: Sample) -> BenchmarkExperimentResult:  # noqa: D102
        _LOGGER.debug(f"Running model experiment (model : {self.model})")
        try:
            module_code = sample.module_code
            module_path = str(sample.module_root / Path(sample.module_name))
            test_cluster = sample.test_cluster

            prompt = TestCaseGenerationPrompt(module_code, module_path)

            model = LLMAgent(
                provider=self.provider, model_name=self.model, temperature=self.temperature
            )
            model.clear_cache()

            llm_query_results = model.query(prompt)
            if llm_query_results is None:
                return BenchmarkExperimentResult(success=False)

            handler = LLMTestCaseHandler(model)
            llm_test_cases_str = handler.extract_test_cases_from_llm_output(llm_query_results)

            deserialize_code_to_testcases(llm_test_cases_str, test_cluster=test_cluster)

            # TODO (Oetgin): Compute coverage using pynguin's coverage computation and return the score in the BenchmarkExperimentResult
            score = compute_coverage(test_cluster, sample.module_name, sample.module_root)

            return BenchmarkExperimentResult(success=True, score=score)
        except BaseException as e:
            _LOGGER.error(f"Error running model experiment (model : {self.model}) : {e}")
            return BenchmarkExperimentResult(success=False)
