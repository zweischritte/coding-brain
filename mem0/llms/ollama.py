import json
import logging
import re
from typing import Any, Dict, List, Optional, Union

try:
    from ollama import Client
except ImportError:
    raise ImportError("The 'ollama' library is required. Please install it using 'pip install ollama'.")

from mem0.configs.llms.base import BaseLlmConfig
from mem0.configs.llms.ollama import OllamaConfig
from mem0.llms.base import LLMBase
from mem0.memory.utils import extract_json

logger = logging.getLogger(__name__)


class OllamaLLM(LLMBase):
    def __init__(self, config: Optional[Union[BaseLlmConfig, OllamaConfig, Dict]] = None):
        # Convert to OllamaConfig if needed
        if config is None:
            config = OllamaConfig()
        elif isinstance(config, dict):
            config = OllamaConfig(**config)
        elif isinstance(config, BaseLlmConfig) and not isinstance(config, OllamaConfig):
            # Convert BaseLlmConfig to OllamaConfig
            config = OllamaConfig(
                model=config.model,
                temperature=config.temperature,
                api_key=config.api_key,
                max_tokens=config.max_tokens,
                top_p=config.top_p,
                top_k=config.top_k,
                enable_vision=config.enable_vision,
                vision_details=config.vision_details,
                http_client_proxies=config.http_client,
            )

        super().__init__(config)

        if not self.config.model:
            self.config.model = "llama3.1:70b"

        self.client = Client(host=self.config.ollama_base_url)

    def _parse_response(self, response, tools):
        """
        Process the response based on whether tools are used or not.

        Args:
            response: The raw response from API.
            tools: The list of tools provided in the request.

        Returns:
            str or dict: The processed response.
        """
        # Get the content from response
        if isinstance(response, dict):
            content = response["message"]["content"]
        else:
            content = response.message.content

        if not tools:
            return content

        processed_response = {
            "content": content,
            "tool_calls": [],
        }

        tool_name = None
        required_keys: List[str] = []
        if tools:
            function_spec = tools[0].get("function", {})
            tool_name = function_spec.get("name")
            required_keys = function_spec.get("parameters", {}).get("required", [])

        if not tool_name:
            return processed_response

        raw_json = extract_json(content)
        raw_json = re.sub(r"<think>.*?</think>", "", raw_json, flags=re.DOTALL).strip()
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            logger.warning("Failed to parse Ollama tool response JSON.")
            return processed_response

        if isinstance(parsed, list):
            if "entities" in required_keys:
                processed_response["tool_calls"].append(
                    {"name": tool_name, "arguments": {"entities": parsed}}
                )
            else:
                for item in parsed:
                    if isinstance(item, dict):
                        processed_response["tool_calls"].append(
                            {"name": tool_name, "arguments": item}
                        )
        elif isinstance(parsed, dict):
            processed_response["tool_calls"].append(
                {"name": tool_name, "arguments": parsed}
            )

        return processed_response

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format=None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
        **kwargs,
    ):
        """
        Generate a response based on the given messages using Ollama.

        Args:
            messages (list): List of message dicts containing 'role' and 'content'.
            response_format (str or object, optional): Format of the response. Defaults to "text".
            tools (list, optional): List of tools that the model can call. Defaults to None.
            tool_choice (str, optional): Tool choice method. Defaults to "auto".
            **kwargs: Additional Ollama-specific parameters.

        Returns:
            str: The generated response.
        """
        request_messages = [dict(msg) for msg in messages]

        # Build parameters for Ollama
        params = {
            "model": self.config.model,
            "messages": request_messages,
        }

        # Handle JSON response format by using Ollama's native format parameter
        if response_format and response_format.get("type") == "json_object":
            params["format"] = "json"
            # Also add JSON format instruction to the last message as a fallback
            if request_messages and request_messages[-1]["role"] == "user":
                request_messages[-1]["content"] += "\n\nPlease respond with valid JSON only."
            else:
                request_messages.append({"role": "user", "content": "Please respond with valid JSON only."})

        if tools and not response_format:
            tool_schema = tools[0].get("function", {}).get("parameters")
            if tool_schema:
                params["format"] = tool_schema
                if request_messages and request_messages[-1]["role"] == "user":
                    request_messages[-1]["content"] += "\n\nRespond with JSON that matches the schema only."
                else:
                    request_messages.append(
                        {"role": "user", "content": "Respond with JSON that matches the schema only."}
                    )

        # Add options for Ollama (temperature, num_predict, top_p)
        options = {
            "temperature": self.config.temperature,
            "num_predict": self.config.max_tokens,
            "top_p": self.config.top_p,
            "top_k": self.config.top_k,
        }
        params["options"] = options

        # Remove OpenAI-specific parameters that Ollama doesn't support
        params.pop("max_tokens", None)  # Ollama uses different parameter names

        response = self.client.chat(**params)
        parsed_response = self._parse_response(response, tools)
        if tools and not parsed_response.get("tool_calls"):
            retry_messages = request_messages + [
                {
                    "role": "user",
                    "content": "Return JSON only and match the schema exactly. No extra text.",
                }
            ]
            params["messages"] = retry_messages
            response = self.client.chat(**params)
            parsed_response = self._parse_response(response, tools)

        return parsed_response
