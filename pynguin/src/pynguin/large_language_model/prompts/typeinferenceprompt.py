#  This file is part of Pynguin.
#
#  SPDX-FileCopyrightText: 2019–2024 Pynguin Contributors
#
#  SPDX-License-Identifier: MIT
#
"""Provides class prompt for type inference using LLMs."""

import textwrap
from collections.abc import Callable
from typing import Any

from pynguin.large_language_model.prompts.base_inference_prompt import BaseInferencePrompt
from pynguin.utils.orderedset import OrderedSet

_ROLE_USER = "<|user|>"


class TypeInferencePrompt(BaseInferencePrompt):
    """Implementation prompt for type inference using LLMs."""

    # TODO: load templates form src/pynguin/resources/ would be cleaner
    def __init__(
        self, callable_obj: Callable[..., Any], subtypes: OrderedSet[str] | None = None
    ) -> None:
        """Creates a new TypeInferencePrompt.

        Args:
            callable_obj: the callable object for which types should be inferred
            subtypes: list of known string subtypes (e.g., "email", "url", etc.)
        """
        super().__init__(callable_obj, subtypes)

    def build_user_prompt(self) -> str:
        """Build the complete prompt for type inference."""
        template = textwrap.dedent(
            """
            You are tasked with inferring parameter types for a given Python function.

            ## Module Context
            - Imports in the module:
            {imports}

            - Parent class name:
            {parent_class}

            - All classes in the same module:
            {all_classes}

            - Known string subtypes:
            {subtype_list}

            ## Target Function
            - Function signature:
            {signature}

            - Docstring:
            {docstring}

            - Function body:
            {body}

            ## Additional Context
            - Other function signatures in the same class:
            {other_functions}

            ## Task
            Infer the parameter types for the target function above.

            Return your answer **only** as JSON in the following format:
            {{
                "param1": <qualname of type>,
                "param2": <qualname of type>
            }}
            """
        ).lstrip()

        formatted_template = template.format(
            parent_class=self._get_parent_class_name(self.callable_obj),
            imports=self._get_imports(self.callable_obj),
            all_classes=self._get_all_classes_in_module(),
            other_functions=self._get_all_function_signatures_in_class(self.callable_obj),
            signature=self._get_signature_str(self.callable_obj),
            docstring=self._get_docstring(self.callable_obj),
            body=self._get_src_code(self.callable_obj),
            subtype_list=self._get_str_subtypes(),
        )
        return f"{formatted_template}"


def get_inference_system_prompt() -> str:
    """Build the system prompt for type inference."""
    return textwrap.dedent(
        """
            You are a Python type inference engine.
            Your task is to analyze given Python functions and infer the parameter types.
            Think step by step. Before inferring types, analyze the given context.
            Reason about each parameter's type based on usage and context.
            Keep this reasoning to yourself and do not include it in the final output.
            Use your knowledge of programming, common libraries, and best practices to infer types.
            Use the provided context to make an informed decision about the types of parameters.
            Always return results in full qualified names, e.g., typing.List[builtins.int].
            *NEVER* use Any or object as a type.
            Only infer types for parameters, exclude self and return types.
            Return your output in JSON format only.

            When a parameter is a string, consider if it matches one of the known
            string subtypes and prefer returning that subtype when appropriate.
            """
    ).strip()
