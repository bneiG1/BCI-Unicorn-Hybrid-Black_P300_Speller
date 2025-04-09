"""
Language Model Module for P300 Speller
====================================
Implements predictive text functionality using n-gram models and word completion
with support for adaptive vocabulary learning.
"""

import numpy as np
from typing import List, Dict, Set
from collections import defaultdict
import pickle
from pathlib import Path
import re

class SpellerLanguageModel:
    """
    Language model for predictive text in P300 speller.
    """
    
    def __init__(self, n_gram_size: int = 3):
        self.n_gram_size = n_gram_size
        self.word_frequencies = defaultdict(int)
        self.n_grams = defaultdict(lambda: defaultdict(int))
        self.vocabulary = set()
        self.user_vocabulary = set()
        self.common_words = set()
        
        # Load common English words
        self._load_common_words()
    
    def _load_common_words(self, filename: str = "common_words.txt"):
        """Load common English words from file."""
        try:
            word_file = Path(__file__).parent / filename
            if word_file.exists():
                with open(word_file, 'r', encoding='utf-8') as f:
                    self.common_words = set(word.strip().lower() for word in f)
        except Exception as e:
            print(f"Warning: Could not load common words: {e}")
    
    def train(self, text: str):
        """
        Train the language model on input text.
        
        Args:
            text: Training text
        """
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Update word frequencies
        for word in words:
            self.word_frequencies[word] += 1
            self.vocabulary.add(word)
        
        # Update n-grams
        padded = ['<START>'] * (self.n_gram_size - 1) + words + ['<END>']
        for i in range(len(words)):
            context = tuple(padded[i:i + self.n_gram_size - 1])
            next_word = padded[i + self.n_gram_size - 1]
            self.n_grams[context][next_word] += 1
    
    def predict_next_char(self, current_text: str, top_k: int = 5) -> List[str]:
        """
        Predict most likely next characters based on current text.
        
        Args:
            current_text: Current input text
            top_k: Number of predictions to return
            
        Returns:
            List of most likely next characters
        """
        # Get current word being typed
        words = current_text.split()
        current_word = words[-1] if words else ""
        
        # Find words that start with current prefix
        matching_words = [word for word in self.vocabulary 
                         if word.startswith(current_word.lower())]
        
        # Get next possible characters
        next_chars = set()
        for word in matching_words:
            if len(word) > len(current_word):
                next_chars.add(word[len(current_word)])
        
        # Score characters based on word frequencies
        char_scores = defaultdict(float)
        for char in next_chars:
            for word in matching_words:
                if len(word) > len(current_word) and word[len(current_word)] == char:
                    char_scores[char] += self.word_frequencies[word]
        
        # Sort by score and return top_k
        sorted_chars = sorted(char_scores.items(), key=lambda x: x[1], reverse=True)
        return [char for char, _ in sorted_chars[:top_k]]
    
    def suggest_completions(self, current_word: str, max_suggestions: int = 3) -> List[str]:
        """
        Suggest word completions for current partial word.
        
        Args:
            current_word: Current partial word
            max_suggestions: Maximum number of suggestions
            
        Returns:
            List of suggested word completions
        """
        if not current_word:
            return []
            
        # Find matching words
        matches = [word for word in self.vocabulary 
                  if word.startswith(current_word.lower())]
        
        # Sort by frequency and user vocabulary preference
        scored_matches = []
        for word in matches:
            score = self.word_frequencies[word]
            if word in self.user_vocabulary:
                score *= 2  # Boost user vocabulary words
            scored_matches.append((word, score))
        
        # Return top suggestions
        sorted_matches = sorted(scored_matches, key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_matches[:max_suggestions]]
    
    def add_to_user_vocabulary(self, word: str):
        """Add word to user's personal vocabulary."""
        self.user_vocabulary.add(word.lower())
        self.vocabulary.add(word.lower())
        self.word_frequencies[word.lower()] += 1
    
    def save_model(self, filepath: str):
        """Save language model to file."""
        model_data = {
            'word_frequencies': dict(self.word_frequencies),
            'n_grams': dict(self.n_grams),
            'vocabulary': self.vocabulary,
            'user_vocabulary': self.user_vocabulary,
            'n_gram_size': self.n_gram_size
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self, filepath: str):
        """Load language model from file."""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.word_frequencies = defaultdict(int, model_data['word_frequencies'])
        self.n_grams = defaultdict(lambda: defaultdict(int), model_data['n_grams'])
        self.vocabulary = model_data['vocabulary']
        self.user_vocabulary = model_data['user_vocabulary']
        self.n_gram_size = model_data['n_gram_size']
