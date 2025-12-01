import json
import os
from typing import Dict, List, Optional

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from pydantic import BaseModel, Field

from akte_classifier.models.prompts import (BasePromptTemplate,
                                            ClassificationPromptTemplate)

load_dotenv()


class ClassificationResult(BaseModel):
    """
    The result of the classification.
    """

    label_codes: List[int] = Field(
        description="The list of applicable label codes (integers) for the text."
    )


class LLMClassifier:
    def __init__(
        self,
        model_name: str,
        descriptions: Dict[int, str],
        prompt_template: Optional[BasePromptTemplate] = None,
        max_length: Optional[int] = None,
    ):
        self.model_name = model_name
        self.descriptions = descriptions
        self.prompt_template = prompt_template or ClassificationPromptTemplate()
        self.max_length = max_length

        # Initialize OpenAI client
        # Supports Nebius or other providers via base_url
        base_url = os.getenv(
            "OPENAI_BASE_URL", "https://api.tokenfactory.nebius.com/v1/"
        )
        api_key = os.getenv("NEBIUS_API_KEY")

        if not api_key:
            logger.warning("No API key found (NEBIUS_API_KEY). LLM calls might fail.")

        self.client = OpenAI(base_url=base_url, api_key=api_key)

    def classify(self, text: str) -> List[int]:
        """
        Classifies the text using the LLM.
        """
        if self.max_length:
            # Heuristic: 1 token ~= 4 chars
            char_limit = self.max_length * 4
            if len(text) > char_limit:
                # logger.warning(f"Truncating text from {len(text)} to {char_limit} chars.")
                text = text[:char_limit]

        # Get code combinations info if available in prompt template
        code_combinations_info = getattr(
            self.prompt_template, "code_combinations_info", None
        )
        prompt = self.prompt_template.format(
            descriptions=self.descriptions,
            text=text,
            code_combinations_info=code_combinations_info,
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that outputs JSON.",
                    },
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )

            content = response.choices[0].message.content
            if not content:
                logger.warning("LLM returned empty content.")
                return []

            # Parse JSON
            try:
                # First try to parse as ClassificationResult directly if the LLM follows schema perfectly
                # But typically we parse the raw JSON first
                data = json.loads(content)
                result = ClassificationResult(**data)
                return result.label_codes
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from LLM: {content}")
                return []
            except Exception as e:
                logger.error(
                    f"Failed to validate Pydantic model: {e}. Content: {content}"
                )
                return []

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return []
