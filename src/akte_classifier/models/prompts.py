from abc import ABC, abstractmethod
from string import Template
from typing import Optional


class BasePromptTemplate(ABC):
    @abstractmethod
    def format(self, **kwargs) -> str:
        pass


class ClassificationPromptTemplate(BasePromptTemplate):
    """
    A template for classification prompts using string.Template for safe substitution.
    Supports dynamic inclusion of code combination patterns and reasoning steps.
    """

    DEFAULT_TEMPLATE = """
You are a legal expert classifier specializing in Dutch property law (kadasterrecht). 
Your task is to classify the following Dutch legal document (akte) into one or more 
of the provided categories based on the PRIMARY legal actions described in the document.

CRITICAL CLASSIFICATION PRINCIPLE:
The CONTENT of the document is LEADING - the title is SUPPORTING only.

IMPORTANT: BE CAREFUL WITH QUOTED TEXT
The document may contain quoted text from other legal documents. When you encounter 
text in quotation marks that follows phrases like "woordelijk vermeld", "woordelijk 
luidend", or "begin citaat", be cautious:
- Such quoted text typically refers to OTHER documents
- However, if the quoted text describes actions that are ALSO being executed in THIS 
  document, those actions are still relevant for classification
- Focus on what THIS document actually does, not what other documents mentioned in 
  quotes do
- When in doubt, prioritize the main body text over quoted references

${code_combinations_info}

ANALYSIS PROCESS (follow these steps in order):
1. READ AND IDENTIFY: Read through the entire document and identify all legal actions mentioned
   - List all actions you find, both primary and secondary
   - Note any quoted text but mark it as potentially from other documents

2. PRIMARY VS SECONDARY: Determine which actions are PRIMARY (actually executed) vs SECONDARY (just mentioned)
   - PRIMARY: Actions that are the main purpose of this document
   - SECONDARY: Actions that are referenced, prerequisites, or incidental mentions
   - Focus on PRIMARY actions for classification

3. MATCH TO CODES: For each PRIMARY action, find the most specific applicable code
   - Match each distinct legal action to a code
   - If multiple codes apply, ensure they represent different aspects of the transaction
   - Consider common code combinations (see above) as context, but don't force them

4. VERIFY DISTINCTNESS: Verify that each code represents a separate, identifiable action
   - Each code should correspond to a distinct legal event in the document
   - Avoid assigning multiple codes for the same action
   - If codes overlap significantly, choose the most specific one

5. TITLE CHECK: Compare with document title (if present)
   - If title aligns with your analysis, this confirms your classification
   - If title conflicts, trust your content-based analysis
   - Title is supporting evidence only, not determinative

6. FINAL VALIDATION: Review your selected codes
   - Do they all represent actions executed in THIS document?
   - Are they all distinct from each other?
   - Do they match the primary purpose of the document?

IMPORTANT GUIDELINES:
- Only assign codes that represent DISTINCT legal actions or events in this specific document
- Do NOT assign codes for actions that are merely mentioned or referenced but not executed
- Focus on the PRIMARY purpose of the document, not secondary or incidental mentions
- Each code should represent a separate, identifiable legal action in the text
- If multiple codes apply, they should represent different aspects of the transaction
- When title and content conflict, the content takes precedence

Here are the available categories (Code: Description):
${descriptions_text}

Text to classify:
\"\"\"
${text}
\"\"\"

Follow the analysis process above step by step. Return the applicable category codes as a 
JSON object with a single key "label_codes" containing a list of integers. If no category 
applies, return an empty list.
"""

    def __init__(self, template_str: str = DEFAULT_TEMPLATE):
        self.template = Template(template_str)

    def format(
        self,
        descriptions: Optional[dict] = None,
        text: Optional[str] = None,
        code_combinations_info: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Formats the prompt with the given descriptions and text.
        
        Args:
            descriptions: Dictionary mapping code IDs to descriptions (can also be passed via kwargs)
            text: The document text to classify (can also be passed via kwargs)
            code_combinations_info: Optional string with information about common code combinations
            **kwargs: Alternative way to pass descriptions and text
        """
        # Support both explicit args and kwargs for backward compatibility
        descriptions = descriptions or kwargs.get("descriptions")
        text = text or kwargs.get("text")
        code_combinations_info = code_combinations_info or kwargs.get("code_combinations_info")
        
        # Also check if code_combinations_info was set as instance attribute
        if code_combinations_info is None:
            code_combinations_info = getattr(self, "code_combinations_info", None)
        
        if not isinstance(descriptions, dict) or not isinstance(text, str):
            raise ValueError(
                "Invalid arguments. Expected 'descriptions' (dict) and 'text' (str)."
            )

        descriptions_text = "\n".join(
            [f"- Code {code}: {desc}" for code, desc in descriptions.items()]
        )
        
        # Default code combinations info if not provided
        if code_combinations_info is None:
            code_combinations_info = (
                "Note: Some codes may occur together frequently in practice. "
                "Use this as context, but base your classification on the actual document content."
            )
        
        return self.template.safe_substitute(
            descriptions_text=descriptions_text,
            text=text,
            code_combinations_info=code_combinations_info,
        )