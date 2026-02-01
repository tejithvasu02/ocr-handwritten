"""
Academic Lexicon and Vocabulary-Constrained Decoding for OCR.
Custom vocabulary for academic, scientific, and business terms.
"""

import re
from typing import List, Set, Dict, Optional, Tuple
import json


# Academic and scientific terms commonly found in handwritten documents
ACADEMIC_LEXICON = {
    # Mathematics
    "theorem", "lemma", "proof", "corollary", "hypothesis", "axiom",
    "equation", "integral", "derivative", "function", "variable", "constant",
    "matrix", "vector", "scalar", "tensor", "eigenvalue", "eigenvector",
    "polynomial", "coefficient", "exponent", "logarithm", "exponential",
    "trigonometric", "sine", "cosine", "tangent", "calculus", "algebra",
    
    # Physics
    "velocity", "acceleration", "momentum", "force", "energy", "power",
    "frequency", "wavelength", "amplitude", "oscillation", "quantum",
    "electromagnetic", "gravitational", "thermodynamics", "entropy",
    "photon", "electron", "proton", "neutron", "nucleus", "atom",
    
    # Chemistry
    "molecule", "compound", "element", "reaction", "catalyst", "oxidation",
    "reduction", "equilibrium", "concentration", "molarity", "solution",
    "precipitate", "titration", "spectroscopy", "chromatography",
    
    # Biology
    "cell", "organism", "protein", "enzyme", "dna", "rna", "gene",
    "chromosome", "mutation", "evolution", "photosynthesis", "metabolism",
    "mitosis", "meiosis", "nucleus", "membrane", "cytoplasm",
    
    # Computer Science
    "algorithm", "complexity", "recursion", "iteration", "function",
    "variable", "array", "string", "integer", "boolean", "object",
    "class", "inheritance", "polymorphism", "encapsulation", "abstraction",
    "database", "network", "protocol", "encryption", "authentication",
    
    # General Academic
    "analysis", "hypothesis", "conclusion", "methodology", "research",
    "experiment", "observation", "data", "result", "discussion",
    "abstract", "introduction", "literature", "reference", "citation",
    "figure", "table", "appendix", "bibliography", "acknowledgment",
    
    # Business Terms
    "revenue", "profit", "loss", "asset", "liability", "equity",
    "dividend", "investment", "portfolio", "budget", "forecast",
    "depreciation", "amortization", "inventory", "receivable", "payable",
}

# Common abbreviations and symbols
ABBREVIATIONS = {
    "e.g.", "i.e.", "etc.", "vs.", "approx.", "avg.", "max.", "min.",
    "fig.", "eq.", "ref.", "ch.", "vol.", "pp.", "no.", "dept.",
    "dr.", "mr.", "ms.", "mrs.", "prof.", "ltd.", "inc.", "corp.",
}

# Greek letters (commonly used in math/science)
GREEK_LETTERS = {
    "alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta",
    "iota", "kappa", "lambda", "mu", "nu", "xi", "omicron", "pi",
    "rho", "sigma", "tau", "upsilon", "phi", "chi", "psi", "omega",
}

# LaTeX commands (for math OCR)
LATEX_COMMANDS = {
    "frac", "sqrt", "sum", "prod", "int", "lim", "log", "ln", "exp",
    "sin", "cos", "tan", "cot", "sec", "csc", "arcsin", "arccos", "arctan",
    "partial", "nabla", "infty", "approx", "neq", "leq", "geq",
    "rightarrow", "leftarrow", "Rightarrow", "Leftarrow",
    "forall", "exists", "in", "notin", "subset", "supset",
    "cup", "cap", "times", "cdot", "div", "pm", "mp",
    "alpha", "beta", "gamma", "delta", "epsilon", "theta", "lambda",
    "mu", "pi", "sigma", "phi", "omega", "Delta", "Sigma", "Pi", "Omega",
}


class AcademicLexicon:
    """
    Manages academic vocabulary for OCR correction and biased decoding.
    """
    
    def __init__(
        self,
        include_academic: bool = True,
        include_greek: bool = True,
        include_latex: bool = True,
        custom_words: Optional[List[str]] = None,
        lexicon_file: Optional[str] = None
    ):
        self.vocabulary: Set[str] = set()
        
        # Build vocabulary
        if include_academic:
            self.vocabulary.update(ACADEMIC_LEXICON)
            self.vocabulary.update(ABBREVIATIONS)
        
        if include_greek:
            self.vocabulary.update(GREEK_LETTERS)
        
        if include_latex:
            self.vocabulary.update(LATEX_COMMANDS)
        
        if custom_words:
            self.vocabulary.update(w.lower() for w in custom_words)
        
        if lexicon_file:
            self.load_from_file(lexicon_file)
        
        # Build prefix tree for efficient lookup
        self._prefix_tree = self._build_prefix_tree()
    
    def load_from_file(self, filepath: str):
        """Load vocabulary from file (JSON or text)."""
        try:
            if filepath.endswith('.json'):
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.vocabulary.update(w.lower() for w in data)
                    elif isinstance(data, dict):
                        self.vocabulary.update(w.lower() for w in data.get('words', []))
            else:
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        word = line.strip().lower()
                        if word:
                            self.vocabulary.update
        except Exception as e:
            print(f"Warning: Could not load lexicon file {filepath}: {e}")
    
    def save_to_file(self, filepath: str):
        """Save vocabulary to file."""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(self.vocabulary)), f, indent=2)
    
    def add_words(self, words: List[str]):
        """Add words to vocabulary."""
        self.vocabulary.update(w.lower() for w in words)
        self._prefix_tree = self._build_prefix_tree()
    
    def _build_prefix_tree(self) -> Dict:
        """Build prefix tree for efficient prefix matching."""
        tree = {}
        for word in self.vocabulary:
            node = tree
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node['$'] = True  # End of word marker
        return tree
    
    def contains(self, word: str) -> bool:
        """Check if word is in vocabulary."""
        return word.lower() in self.vocabulary
    
    def get_prefix_matches(self, prefix: str, max_results: int = 10) -> List[str]:
        """Get words starting with prefix."""
        prefix = prefix.lower()
        matches = [w for w in self.vocabulary if w.startswith(prefix)]
        return sorted(matches)[:max_results]
    
    def get_similar(self, word: str, max_distance: int = 2) -> List[str]:
        """Get similar words within edit distance."""
        word = word.lower()
        similar = []
        
        for vocab_word in self.vocabulary:
            dist = self._edit_distance(word, vocab_word)
            if dist <= max_distance:
                similar.append((vocab_word, dist))
        
        return [w for w, d in sorted(similar, key=lambda x: x[1])]
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """Compute Levenshtein edit distance."""
        if len(s1) < len(s2):
            return AcademicLexicon._edit_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]


class VocabularyConstrainedDecoder:
    """
    Vocabulary-constrained decoding for OCR.
    Biases model output towards known vocabulary.
    """
    
    def __init__(
        self,
        lexicon: AcademicLexicon,
        bias_weight: float = 0.3,
        oov_penalty: float = 0.1
    ):
        """
        Args:
            lexicon: AcademicLexicon instance
            bias_weight: Weight for vocabulary bias (0-1)
            oov_penalty: Penalty for out-of-vocabulary words (0-1)
        """
        self.lexicon = lexicon
        self.bias_weight = bias_weight
        self.oov_penalty = oov_penalty
    
    def score_candidate(self, text: str) -> Tuple[float, Dict]:
        """
        Score a candidate text based on vocabulary coverage.
        
        Returns:
            Tuple of (score, details)
        """
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        
        if not words:
            return 1.0, {'in_vocab': 0, 'out_vocab': 0, 'total': 0}
        
        in_vocab = sum(1 for w in words if self.lexicon.contains(w))
        out_vocab = len(words) - in_vocab
        
        # Calculate score
        vocab_ratio = in_vocab / len(words)
        score = vocab_ratio * self.bias_weight + (1 - self.bias_weight)
        score -= out_vocab * self.oov_penalty / len(words)
        
        return max(0.0, min(1.0, score)), {
            'in_vocab': in_vocab,
            'out_vocab': out_vocab,
            'total': len(words),
            'vocab_ratio': vocab_ratio
        }
    
    def rerank_hypotheses(
        self,
        hypotheses: List[Tuple[str, float]]
    ) -> List[Tuple[str, float, Dict]]:
        """
        Rerank beam search hypotheses using vocabulary scores.
        
        Args:
            hypotheses: List of (text, model_score) tuples
        
        Returns:
            Reranked list of (text, combined_score, details) tuples
        """
        scored = []
        
        for text, model_score in hypotheses:
            vocab_score, details = self.score_candidate(text)
            combined_score = (1 - self.bias_weight) * model_score + self.bias_weight * vocab_score
            scored.append((text, combined_score, details))
        
        return sorted(scored, key=lambda x: x[1], reverse=True)
    
    def correct_with_vocabulary(
        self,
        text: str,
        max_corrections: int = 5
    ) -> str:
        """
        Correct text using vocabulary lookup.
        
        Args:
            text: Input text
            max_corrections: Maximum number of corrections
        
        Returns:
            Corrected text
        """
        words = text.split()
        corrected = []
        corrections_made = 0
        
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
            
            if word and not self.lexicon.contains(word) and corrections_made < max_corrections:
                similar = self.lexicon.get_similar(word, max_distance=1)
                if similar:
                    # Preserve case
                    correction = similar[0]
                    if word.isupper():
                        correction = correction.upper()
                    elif word[0].isupper():
                        correction = correction.capitalize()
                    word = correction
                    corrections_made += 1
            
            corrected.append(prefix + word + suffix)
        
        return ' '.join(corrected)


# Pre-built lexicon instance
DEFAULT_LEXICON = AcademicLexicon()
