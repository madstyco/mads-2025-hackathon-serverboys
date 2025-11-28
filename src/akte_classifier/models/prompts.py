from abc import ABC, abstractmethod
from string import Template


class BasePromptTemplate(ABC):
    @abstractmethod
    def format(self, **kwargs) -> str:
        pass


class ClassificationPromptTemplate(BasePromptTemplate):
    """
    A template for classification prompts using string.Template for safe substitution.
    """

    DEFAULT_TEMPLATE = """
You are a legal expert classifier. Your task is to classify the following Dutch legal text into one or more of the provided categories.

Here are the available categories (Code: Description):
${descriptions_text}

Text to classify:
\"\"\"
${text}
\"\"\"

Analyze the text carefully. Return the applicable category codes as a JSON object with a single key "label_codes" containing a list of integers.
If no category applies, return an empty list.
"""

    def __init__(self, template_str: str = DEFAULT_TEMPLATE):
        self.template = Template(template_str)

    def format(self, **kwargs) -> str:
        """
        Formats the prompt with the given descriptions and text.
        """
        descriptions = kwargs.get("descriptions")
        text = kwargs.get("text")

        if not isinstance(descriptions, dict) or not isinstance(text, str):
            raise ValueError(
                "Invalid arguments. Expected 'descriptions' (dict) and 'text' (str)."
            )

        descriptions_text = "\n".join(
            [f"- Code {code}: {desc}" for code, desc in descriptions.items()]
        )
        return self.template.safe_substitute(
            descriptions_text=descriptions_text, text=text
        )
