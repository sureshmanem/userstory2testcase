"""
ModelManualTestLLM.py
Implements the LLM class for interacting with Azure OpenAI.
"""

import os
import time
import configparser
from typing import Dict, Any, Optional, Callable
from openai import AzureOpenAI
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate


class LLM:
    """
    Azure OpenAI LLM client for generating manual tests.
    Reads configuration from environment variables and Config.properties.
    """

    def __init__(self):
        """Initialize the Azure OpenAI client with environment variables and config."""
        # Load environment variables from .env file if present
        load_dotenv()

        # Read temperature from Config.properties
        config = configparser.ConfigParser()
        config_path = os.path.join("Config", "Config.properties")
        config.read(config_path)

        self.temperature = float(config.get("LLM", "TEMPERATURE", fallback="0.05"))

        # Read Azure OpenAI configuration from environment variables
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION")
        self.deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
        self.model_name = os.getenv("AZURE_OPENAI_MODEL_NAME", "gpt-35-turbo")

        # Validate required environment variables
        missing_vars = []
        if not self.api_key:
            missing_vars.append("AZURE_OPENAI_API_KEY")
        if not self.endpoint:
            missing_vars.append("AZURE_OPENAI_ENDPOINT")
        if not self.api_version:
            missing_vars.append("AZURE_OPENAI_API_VERSION")
        if not self.deployment_name:
            missing_vars.append("AZURE_OPENAI_DEPLOYMENT_NAME")

        if missing_vars:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing_vars)}\n"
                f"Please set these in your .env file or environment.\n"
                f"See .env.example for reference."
            )

        # Initialize Azure OpenAI client
        try:
            self.client = AzureOpenAI(
                api_key=self.api_key,
                api_version=self.api_version,
                azure_endpoint=self.endpoint
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Azure OpenAI client: {str(e)}")

    def get_azure_client_info(self) -> Dict[str, Any]:
        """
        Return non-secret configuration information.

        Returns:
            Dict with endpoint, version, deployment, model, and temperature.
        """
        return {
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "deployment_name": self.deployment_name,
            "model_name": self.model_name,
            "temperature": self.temperature
        }

    def ping(self) -> bool:
        """
        Validate connectivity to Azure OpenAI by making a minimal request.

        Returns:
            True if connection successful.

        Raises:
            Exception if connection fails.
        """
        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=self.temperature
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI connection failed: {str(e)}")

    def send_request(
        self,
        template_prompt: str,
        input_variables: list,
        input_variables_dict: Dict[str, str],
        prompt_callback: Optional[Callable[[str], None]] = None,
        metrics_callback: Optional[Callable[[Dict[str, Any]], None]] = None
    ) -> str:
        """
        Send a request to Azure OpenAI using a prompt template.

        Args:
            template_prompt: LangChain prompt template string.
            input_variables: List of variable names in the template.
            input_variables_dict: Dictionary mapping variable names to values.
            prompt_callback: Optional callback to receive the rendered prompt.
            metrics_callback: Optional callback to receive metrics dict.

        Returns:
            The response text from the LLM.
        """
        # Render the prompt using LangChain PromptTemplate
        prompt_template = PromptTemplate(
            input_variables=input_variables,
            template=template_prompt
        )
        rendered_prompt = prompt_template.format(**input_variables_dict)

        # Call the prompt callback if provided
        if prompt_callback:
            prompt_callback(rendered_prompt)

        # Measure latency
        start_time = time.time()

        try:
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=[{"role": "user", "content": rendered_prompt}],
                temperature=self.temperature
            )
        except Exception as e:
            raise RuntimeError(f"Azure OpenAI API request failed: {str(e)}")

        end_time = time.time()
        latency_seconds = end_time - start_time

        # Extract response content
        response_text = response.choices[0].message.content

        # Extract usage metrics if available
        metrics = {
            "latency_seconds": latency_seconds,
            "response_model": getattr(response, "model", self.model_name),
        }

        if hasattr(response, "usage") and response.usage:
            metrics["input_tokens"] = getattr(response.usage, "prompt_tokens", 0)
            metrics["output_tokens"] = getattr(response.usage, "completion_tokens", 0)
            metrics["total_tokens"] = getattr(response.usage, "total_tokens", 0)
        else:
            metrics["input_tokens"] = 0
            metrics["output_tokens"] = 0
            metrics["total_tokens"] = 0

        # Calculate output word and character counts
        metrics["output_words"] = len(response_text.split())
        metrics["output_chars"] = len(response_text)

        # Call the metrics callback if provided
        if metrics_callback:
            metrics_callback(metrics)

        return response_text
