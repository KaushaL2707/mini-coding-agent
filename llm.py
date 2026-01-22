"""
LLM wrapper module.

STEP 5: Interface to various LLM providers (OpenAI, Anthropic, Groq).
"""
from abc import ABC, abstractmethod
from typing import Optional

from config import (
    LLM_PROVIDER,
    LLM_MODELS,
    OPENAI_API_KEY,
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
)


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Generate a response from the LLM."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model name being used."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI GPT provider."""
    
    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or OPENAI_API_KEY
        self._model = model or LLM_MODELS.get("openai", "gpt-4o-mini")
        self.client = None
        
    def _ensure_client(self):
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install openai: pip install openai")
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._ensure_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
            max_tokens=4000,
        )
        
        return response.choices[0].message.content


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider."""
    
    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or ANTHROPIC_API_KEY
        self._model = model or LLM_MODELS.get("anthropic", "claude-3-haiku-20240307")
        self.client = None
    
    def _ensure_client(self):
        if self.client is None:
            try:
                import anthropic
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install anthropic: pip install anthropic")
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._ensure_client()
        
        kwargs = {
            "model": self._model,
            "max_tokens": 4000,
            "messages": [{"role": "user", "content": prompt}],
        }
        
        if system_prompt:
            kwargs["system"] = system_prompt
        
        response = self.client.messages.create(**kwargs)
        
        return response.content[0].text


class GroqProvider(LLMProvider):
    """Groq (fast inference) provider."""
    
    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or GROQ_API_KEY
        self._model = model or LLM_MODELS.get("groq", "llama-3.1-8b-instant")
        self.client = None
    
    def _ensure_client(self):
        if self.client is None:
            try:
                from groq import Groq
                self.client = Groq(api_key=self.api_key)
            except ImportError:
                raise ImportError("Please install groq: pip install groq")
    
    @property
    def model_name(self) -> str:
        return self._model
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        self._ensure_client()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.client.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=0.2,
            max_tokens=4000,
        )
        
        return response.choices[0].message.content


def get_llm_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """
    Factory function to get the appropriate LLM provider.
    
    Args:
        provider_name: "openai", "anthropic", or "groq"
        
    Returns:
        LLMProvider instance
    """
    name = (provider_name or LLM_PROVIDER).lower()
    
    providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "groq": GroqProvider,
    }
    
    if name not in providers:
        raise ValueError(f"Unknown provider: {name}. Choose from: {list(providers.keys())}")
    
    return providers[name]()


class LLM:
    """
    Main LLM interface for the coding agent.
    Provides high-level methods for code-related tasks.
    """
    
    SYSTEM_PROMPT = """You are a senior software engineer and expert code reviewer.
You analyze code carefully and provide precise, actionable suggestions.
When suggesting fixes:
1. Explain the issue clearly
2. Show the exact code changes needed
3. Use diff format when showing changes
4. Consider edge cases and potential side effects

Be concise but thorough. Focus on the specific task at hand."""
    
    def __init__(self, provider: Optional[LLMProvider] = None):
        self.provider = provider or get_llm_provider()
        print(f"🤖 Using LLM: {self.provider.model_name}")
    
    def analyze_code(self, code_context: str, user_prompt: str) -> str:
        """
        Analyze code and respond to user's request.
        
        Args:
            code_context: Relevant code from the repository
            user_prompt: User's question or task
            
        Returns:
            LLM's analysis and suggestions
        """
        prompt = f"""## Relevant Code from Repository

{code_context}

## User Request

{user_prompt}

## Instructions

Analyze the code above and respond to the user's request.
If suggesting code changes, show them in a clear diff format.
"""
        
        return self.provider.generate(prompt, self.SYSTEM_PROMPT)
    
    def suggest_fix(self, code_context: str, issue_description: str) -> str:
        """
        Suggest a fix for a described issue.
        
        Args:
            code_context: Relevant code
            issue_description: Description of the bug/issue
            
        Returns:
            Suggested fix with explanation
        """
        prompt = f"""## Problem

{issue_description}

## Relevant Code

{code_context}

## Task

1. Identify the root cause of the issue
2. Explain why this is happening
3. Provide the exact fix in diff format
4. Note any potential side effects of the fix
"""
        
        return self.provider.generate(prompt, self.SYSTEM_PROMPT)
    
    def generate_patch(self, code_context: str, instructions: str) -> str:
        """
        Generate a patch/diff for requested changes.
        
        Args:
            code_context: Current code
            instructions: What changes to make
            
        Returns:
            Unified diff format patch
        """
        patch_prompt = """You are a code patching assistant. 
Output ONLY the changes in unified diff format.
Do not include explanations outside the diff."""
        
        prompt = f"""Generate a patch for the following request.

## Current Code

{code_context}

## Requested Changes

{instructions}

Output the changes as a unified diff (--- a/file, +++ b/file format).
"""
        
        return self.provider.generate(prompt, patch_prompt)


if __name__ == "__main__":
    # Test LLM connection
    llm = LLM()
    
    test_code = '''
def calculate_average(numbers):
    total = 0
    for n in numbers:
        total += n
    return total / len(numbers)
'''
    
    response = llm.analyze_code(
        test_code,
        "This function crashes when the list is empty. How do I fix it?"
    )
    
    print("\n" + "="*60)
    print("LLM Response:")
    print("="*60)
    print(response)
