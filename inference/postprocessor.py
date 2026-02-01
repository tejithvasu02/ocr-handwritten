"""
Language Model Post-Processor for OCR Output Correction.
Provides spell-checking, grammar correction, and semantic plausibility filtering.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class CorrectionResult:
    """Result from post-processing correction."""
    original: str
    corrected: str
    confidence: float
    corrections: List[Dict]
    is_plausible: bool


class SpellChecker:
    """
    Spell checker for OCR output correction.
    Uses pyspellchecker or falls back to basic correction.
    """
    
    def __init__(self, language: str = "en", custom_words: Optional[List[str]] = None):
        self.language = language
        self.custom_words = set(custom_words or [])
        self.spell = None
        
        try:
            from spellchecker import SpellChecker as PySpellChecker
            self.spell = PySpellChecker(language=language)
            if custom_words:
                self.spell.word_frequency.load_words(custom_words)
        except ImportError:
            print("Warning: pyspellchecker not installed. Using basic correction.")
    
    def check(self, text: str) -> List[Dict]:
        """
        Check text for spelling errors.
        
        Returns:
            List of dicts with 'word', 'suggestions', 'position'
        """
        if self.spell is None:
            return []
        
        words = text.split()
        errors = []
        
        position = 0
        for word in words:
            # Clean word for checking
            clean_word = re.sub(r'[^\w]', '', word.lower())
            
            if clean_word and clean_word not in self.custom_words:
                if self.spell.unknown([clean_word]):
                    suggestions = list(self.spell.candidates(clean_word) or [])[:5]
                    errors.append({
                        'word': word,
                        'clean_word': clean_word,
                        'suggestions': suggestions,
                        'position': position
                    })
            
            position += len(word) + 1
        
        return errors
    
    def correct(self, text: str, max_edit_distance: int = 2) -> str:
        """
        Auto-correct spelling errors.
        
        Args:
            text: Input text
            max_edit_distance: Maximum edit distance for corrections
        
        Returns:
            Corrected text
        """
        if self.spell is None:
            return text
        
        words = text.split()
        corrected_words = []
        
        for word in words:
            # Preserve punctuation
            prefix = ""
            suffix = ""
            
            while word and not word[0].isalnum():
                prefix += word[0]
                word = word[1:]
            
            while word and not word[-1].isalnum():
                suffix = word[-1] + suffix
                word = word[:-1]
            
            if word and word.lower() not in self.custom_words:
                if self.spell.unknown([word.lower()]):
                    correction = self.spell.correction(word.lower())
                    if correction and correction != word.lower():
                        # Preserve original case
                        if word.isupper():
                            correction = correction.upper()
                        elif word[0].isupper():
                            correction = correction.capitalize()
                        word = correction
            
            corrected_words.append(prefix + word + suffix)
        
        return ' '.join(corrected_words)


class GrammarChecker:
    """
    Basic grammar checker for OCR output.
    Uses simple rules or language_tool_python if available.
    """
    
    def __init__(self, language: str = "en-US"):
        self.language = language
        self.tool = None
        
        try:
            import language_tool_python
            self.tool = language_tool_python.LanguageTool(language)
        except ImportError:
            print("Warning: language_tool_python not installed. Using rule-based correction.")
    
    def check(self, text: str) -> List[Dict]:
        """Check text for grammar errors."""
        if self.tool is None:
            return self._rule_based_check(text)
        
        matches = self.tool.check(text)
        return [
            {
                'message': m.message,
                'context': m.context,
                'suggestions': m.replacements[:3],
                'offset': m.offset,
                'length': m.errorLength,
                'rule_id': m.ruleId
            }
            for m in matches
        ]
    
    def _rule_based_check(self, text: str) -> List[Dict]:
        """Simple rule-based grammar checking."""
        errors = []
        
        # Common OCR-induced grammar issues
        patterns = [
            (r'\b([Ii])\s+([a-z])', "Missing uppercase after 'I'"),
            (r'\s{2,}', "Multiple spaces"),
            (r'(\w)\.\s*([a-z])', "Missing uppercase after period"),
            (r'\b(teh|hte|adn|nad)\b', "Common misspelling"),
        ]
        
        for pattern, message in patterns:
            for match in re.finditer(pattern, text):
                errors.append({
                    'message': message,
                    'context': text[max(0, match.start()-10):match.end()+10],
                    'offset': match.start(),
                    'length': match.end() - match.start()
                })
        
        return errors
    
    def correct(self, text: str) -> str:
        """Apply grammar corrections."""
        if self.tool is None:
            return self._rule_based_correct(text)
        
        return self.tool.correct(text)
    
    def _rule_based_correct(self, text: str) -> str:
        """Apply simple rule-based corrections."""
        # Fix multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)
        
        # Fix common OCR errors
        corrections = {
            r'\bteh\b': 'the',
            r'\bhte\b': 'the',
            r'\badn\b': 'and',
            r'\bnad\b': 'and',
            r'\bwiht\b': 'with',
            r'\bform\b(?=\s+the)': 'from',
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text


class SemanticFilter:
    """
    Semantic plausibility filter for OCR output.
    Rejects or flags outputs that don't make semantic sense.
    """
    
    def __init__(self, min_word_ratio: float = 0.5, min_alpha_ratio: float = 0.7):
        """
        Args:
            min_word_ratio: Minimum ratio of dictionary words to total words
            min_alpha_ratio: Minimum ratio of alphabetic characters
        """
        self.min_word_ratio = min_word_ratio
        self.min_alpha_ratio = min_alpha_ratio
        self.vocab = set()
        
        try:
            from spellchecker import SpellChecker
            spell = SpellChecker()
            self.vocab = spell.word_frequency.dictionary
        except ImportError:
            pass
    
    def is_plausible(self, text: str) -> Tuple[bool, float, Dict]:
        """
        Check if text is semantically plausible.
        
        Returns:
            Tuple of (is_plausible, confidence, details)
        """
        if not text.strip():
            return False, 0.0, {'reason': 'empty_text'}
        
        # Check alpha ratio
        alpha_count = sum(1 for c in text if c.isalpha())
        total_count = len(text.replace(' ', ''))
        
        if total_count == 0:
            return False, 0.0, {'reason': 'no_characters'}
        
        alpha_ratio = alpha_count / total_count
        
        if alpha_ratio < self.min_alpha_ratio:
            return False, alpha_ratio, {
                'reason': 'low_alpha_ratio',
                'alpha_ratio': alpha_ratio
            }
        
        # Check word ratio (if vocab available)
        if self.vocab:
            words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
            if words:
                valid_words = sum(1 for w in words if w in self.vocab)
                word_ratio = valid_words / len(words)
                
                if word_ratio < self.min_word_ratio:
                    return False, word_ratio, {
                        'reason': 'low_word_ratio',
                        'word_ratio': word_ratio,
                        'valid_words': valid_words,
                        'total_words': len(words)
                    }
        
        # Check for repetition patterns (common OCR failure)
        if self._has_excessive_repetition(text):
            return False, 0.3, {'reason': 'excessive_repetition'}
        
        # Passed all checks
        confidence = min(1.0, alpha_ratio * 1.2)
        return True, confidence, {'reason': 'passed'}
    
    def _has_excessive_repetition(self, text: str, threshold: int = 3) -> bool:
        """Check for excessive character or word repetition."""
        # Check for repeated characters (e.g., "aaaaaa")
        if re.search(r'(.)\1{4,}', text):
            return True
        
        # Check for repeated words (e.g., "the the the")
        words = text.lower().split()
        if len(words) >= 3:
            for i in range(len(words) - threshold + 1):
                if len(set(words[i:i+threshold])) == 1:
                    return True
        
        return False


class OCRPostProcessor:
    """
    Complete post-processing pipeline for OCR output.
    Combines spell-checking, grammar correction, and semantic filtering.
    """
    
    def __init__(
        self,
        enable_spell_check: bool = True,
        enable_grammar: bool = True,
        enable_semantic_filter: bool = True,
        custom_vocabulary: Optional[List[str]] = None,
        language: str = "en"
    ):
        self.enable_spell_check = enable_spell_check
        self.enable_grammar = enable_grammar
        self.enable_semantic_filter = enable_semantic_filter
        
        self.spell_checker = SpellChecker(language, custom_vocabulary) if enable_spell_check else None
        self.grammar_checker = GrammarChecker(f"{language}-US") if enable_grammar else None
        self.semantic_filter = SemanticFilter() if enable_semantic_filter else None
    
    def process(
        self,
        text: str,
        auto_correct: bool = True,
        reject_implausible: bool = False
    ) -> CorrectionResult:
        """
        Process OCR output with full correction pipeline.
        
        Args:
            text: Raw OCR output
            auto_correct: Automatically apply corrections
            reject_implausible: Return empty string for implausible outputs
        
        Returns:
            CorrectionResult with corrections and metadata
        """
        original = text
        corrected = text
        corrections = []
        is_plausible = True
        confidence = 1.0
        
        # Step 1: Semantic filtering
        if self.semantic_filter:
            is_plausible, confidence, details = self.semantic_filter.is_plausible(text)
            
            if not is_plausible and reject_implausible:
                return CorrectionResult(
                    original=original,
                    corrected="",
                    confidence=confidence,
                    corrections=[{'type': 'rejected', 'reason': details['reason']}],
                    is_plausible=False
                )
        
        # Step 2: Spell checking
        if self.spell_checker and auto_correct:
            spell_errors = self.spell_checker.check(corrected)
            corrections.extend([{'type': 'spelling', **e} for e in spell_errors])
            corrected = self.spell_checker.correct(corrected)
        
        # Step 3: Grammar correction
        if self.grammar_checker and auto_correct:
            grammar_errors = self.grammar_checker.check(corrected)
            corrections.extend([{'type': 'grammar', **e} for e in grammar_errors])
            corrected = self.grammar_checker.correct(corrected)
        
        return CorrectionResult(
            original=original,
            corrected=corrected,
            confidence=confidence,
            corrections=corrections,
            is_plausible=is_plausible
        )
    
    def batch_process(
        self,
        texts: List[str],
        **kwargs
    ) -> List[CorrectionResult]:
        """Process multiple OCR outputs."""
        return [self.process(text, **kwargs) for text in texts]
