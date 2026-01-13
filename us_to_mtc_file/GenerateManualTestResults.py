"""
GenerateManualTestResults.py
Implements the ManualTestGenerator class for building prompts and calling LLM.
"""

import configparser
from typing import List, Optional, Callable, Dict, Any
from us_to_mtc_file.ModelManualTestLLM import LLM


class ManualTestGenerator:
    """
    Generates manual test cases from user stories using Azure OpenAI.
    Reads prompt templates from ConfigGPT.properties.
    """

    def __init__(self, llm_family: str = "GPT"):
        """
        Initialize the generator.

        Args:
            llm_family: LLM family to use (e.g., "GPT").
        """
        self.llm_family = llm_family
        self.llm = LLM()

        # Load prompt configuration
        self.config = configparser.ConfigParser()
        self.config.read("Config/ConfigGPT.properties")

        # Load prompt fragments
        self._load_prompt_fragments()

    def _load_prompt_fragments(self):
        """Load all prompt fragments from configuration."""
        # AdditionalAcceptanceCriteria
        self.acceptance_criteria_prompt = self.config.get("AdditionalAcceptanceCriteria", "acceptance_criteria_prompt")
        self.input_user_story = self.config.get("AdditionalAcceptanceCriteria", "input_user_story")

        # Prompt instructions
        self.instructions = self.config.get("Prompt", "instructions")
        self.reuse_instruction = self.config.get("Prompt", "reuse_instruction")
        self.both_instruction = self.config.get("Prompt", "both_instruction")
        self.positive_instruction = self.config.get("Prompt", "positive_instruction")
        self.negative_instruction = self.config.get("Prompt", "negative_instruction")
        self.additional_intelligence_instructions = self.config.get("Prompt", "additional_intelligence_instructions")

        # Context prompts
        self.manual_test_prompt = self.config.get("Context", "manual_test_prompt")
        self.manual_test_context = self.config.get("Context", "manual_test_context")
        self.manual_test_prompt_instruction = self.config.get("Context", "manual_test_prompt_instruction")
        self.manual_test_instruction = self.config.get("Context", "manual_test_instruction")
        self.manual_test_userStory_instruction = self.config.get("Context", "manual_test_userStory_instruction")
        self.manual_test_userStory = self.config.get("Context", "manual_test_userStory")

        # NoContext prompts
        self.manual_test_noContext_prompt = self.config.get("NoContext", "manual_test_noContext_prompt")
        self.manual_test_noContext_instruction = self.config.get("NoContext", "manual_test_noContext_instruction")
        self.manual_test_noContext_instruction_userStory = self.config.get("NoContext", "manual_test_noContext_instruction_userStory")
        self.manual_test_noContext_userStory = self.config.get("NoContext", "manual_test_noContext_userStory")

    def _build_instruction(self, test_type: str, enable_additional_intelligence: Optional[bool]) -> str:
        """
        Build instruction string based on test type.

        Args:
            test_type: "Both", "Positive", or "Negative".
            enable_additional_intelligence: If False, append additional intelligence instructions.

        Returns:
            Complete instruction string.
        """
        # Base instructions
        instruction = self.instructions + "\n" + self.reuse_instruction + "\n"

        # Add test type specific instruction
        if test_type.lower() == "both":
            instruction += self.both_instruction
        elif test_type.lower() == "positive":
            instruction += self.positive_instruction
        elif test_type.lower() == "negative":
            instruction += self.negative_instruction
        else:
            instruction += self.both_instruction  # Default to both

        # Add additional intelligence instruction if disabled
        if enable_additional_intelligence is False:
            instruction += "\n" + self.additional_intelligence_instructions

        return instruction

    def _generate_additional_acceptance_criteria(
        self,
        user_story: str,
        prompt_callback: Optional[Callable[[str], None]] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """
        Generate additional acceptance criteria for a user story.

        Args:
            user_story: User story text.
            prompt_callback: Optional callback for rendered prompt.
            metrics_callback: Optional callback for metrics.

        Returns:
            Generated acceptance criteria text.
        """
        template = self.acceptance_criteria_prompt + "\n" + self.input_user_story
        input_vars = ["UserStory"]
        input_dict = {"UserStory": user_story}

        result = self.llm.send_request(
            template,
            input_vars,
            input_dict,
            prompt_callback=prompt_callback,
            metrics_callback=metrics_callback
        )

        return result

    def get_manual_test_no_context(
        self,
        test_type: str,
        input_length: int,
        query: str,
        enable_additional_intelligence: Optional[bool] = None,
        generate_additional_acceptance_criteria: Optional[bool] = None,
        auto_send_to_llm: Optional[bool] = None,
        prompt_callback: Optional[Callable[[str], None]] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[str]:
        """
        Generate manual test without context.

        Args:
            test_type: "Both", "Positive", or "Negative".
            input_length: Number of input user stories (for logging).
            query: User story text.
            enable_additional_intelligence: If None, prompts user (CLI mode).
            generate_additional_acceptance_criteria: If None, prompts user (CLI mode).
            auto_send_to_llm: If True, skip confirmation prompts (UI mode).
            prompt_callback: Optional callback for rendered prompt.
            metrics_callback: Optional callback for metrics.

        Returns:
            List containing the generated manual test text.
        """
        user_story = query

        # Handle additional acceptance criteria
        if generate_additional_acceptance_criteria is None and auto_send_to_llm is None:
            # CLI mode - prompt user
            gen_ac = input("Do you want to generate additional acceptance criteria? (Yes/No): ").strip().lower()
            generate_additional_acceptance_criteria = gen_ac in ["yes", "y"]

        if generate_additional_acceptance_criteria:
            print("Generating additional acceptance criteria...")
            additional_ac = self._generate_additional_acceptance_criteria(
                user_story,
                prompt_callback=prompt_callback,
                metrics_callback=metrics_callback
            )
            user_story = user_story + "\n\nAdditional Acceptance Criteria:\n" + additional_ac

        # Handle additional intelligence
        if enable_additional_intelligence is None and auto_send_to_llm is None:
            # CLI mode - prompt user
            enable_ai = input("Do you want to enable additional intelligence? (Yes/No): ").strip().lower()
            enable_additional_intelligence = enable_ai in ["yes", "y"]

        # Build instruction
        instruction = self._build_instruction(test_type, enable_additional_intelligence)

        # Build final prompt
        template = (
            self.manual_test_noContext_prompt + "\n" +
            self.manual_test_noContext_instruction + "\n" +
            self.manual_test_noContext_instruction_userStory + "\n" +
            self.manual_test_noContext_userStory
        )

        input_vars = ["instruction", "UserStory"]
        input_dict = {
            "instruction": instruction,
            "UserStory": user_story
        }

        # Send request
        result = self.llm.send_request(
            template,
            input_vars,
            input_dict,
            prompt_callback=prompt_callback,
            metrics_callback=metrics_callback
        )

        return [result]

    def get_manual_test(
        self,
        test_type: str,
        input_length: int,
        query: str,
        context: str,
        enable_additional_intelligence: Optional[bool] = None,
        generate_additional_acceptance_criteria: Optional[bool] = None,
        auto_send_to_llm: Optional[bool] = None,
        prompt_callback: Optional[Callable[[str], None]] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> List[str]:
        """
        Generate manual test with context.

        Args:
            test_type: "Both", "Positive", or "Negative".
            input_length: Number of input user stories (for logging).
            query: User story text.
            context: Retrieved context text.
            enable_additional_intelligence: If None, prompts user (CLI mode).
            generate_additional_acceptance_criteria: If None, prompts user (CLI mode).
            auto_send_to_llm: If True, skip confirmation prompts (UI mode).
            prompt_callback: Optional callback for rendered prompt.
            metrics_callback: Optional callback for metrics.

        Returns:
            List containing the generated manual test text.
        """
        user_story = query

        # Handle additional acceptance criteria
        if generate_additional_acceptance_criteria is None and auto_send_to_llm is None:
            # CLI mode - prompt user
            gen_ac = input("Do you want to generate additional acceptance criteria? (Yes/No): ").strip().lower()
            generate_additional_acceptance_criteria = gen_ac in ["yes", "y"]

        if generate_additional_acceptance_criteria:
            print("Generating additional acceptance criteria...")
            additional_ac = self._generate_additional_acceptance_criteria(
                user_story,
                prompt_callback=prompt_callback,
                metrics_callback=metrics_callback
            )
            user_story = user_story + "\n\nAdditional Acceptance Criteria:\n" + additional_ac

        # Handle additional intelligence
        if enable_additional_intelligence is None and auto_send_to_llm is None:
            # CLI mode - prompt user
            enable_ai = input("Do you want to enable additional intelligence? (Yes/No): ").strip().lower()
            enable_additional_intelligence = enable_ai in ["yes", "y"]

        # Build instruction
        instruction = self._build_instruction(test_type, enable_additional_intelligence)

        # Build final prompt with context
        template = (
            self.manual_test_prompt + "\n" +
            self.manual_test_context + "\n" +
            self.manual_test_prompt_instruction + "\n" +
            self.manual_test_instruction + "\n" +
            self.manual_test_userStory_instruction + "\n" +
            self.manual_test_userStory
        )

        input_vars = ["context", "instruction", "UserStory"]
        input_dict = {
            "context": context,
            "instruction": instruction,
            "UserStory": user_story
        }

        # Send request
        result = self.llm.send_request(
            template,
            input_vars,
            input_dict,
            prompt_callback=prompt_callback,
            metrics_callback=metrics_callback
        )

        return [result]
