import openai
from typing import List

class LanguageModel:
    """
    Wrapper for querying GPT-3.5 or similar LLMs for word prediction in BCI context.
    """
    def __init__(self, api_key: str, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key
        self.model = model
        openai.api_key = api_key

    def predict_words(self, context: str, n_suggestions: int = 3) -> List[str]:
        """
        Query the LLM for next word predictions given the current context.
        Args:
            context: Current text (sentence/phrase)
            n_suggestions: Number of word suggestions to return
        Returns:
            List of predicted next words/phrases
        """
        prompt = (
            f"Given the partial sentence: '{context}', suggest the {n_suggestions} most likely next words or short phrases. "
            "Return only the suggestions, comma-separated."
        )
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16 * n_suggestions,
            n=1,
            stop=None,
            temperature=0.7,
        )
        text = response.choices[0].message['content']
        suggestions = [w.strip() for w in text.split(',') if w.strip()]
        return suggestions[:n_suggestions]
