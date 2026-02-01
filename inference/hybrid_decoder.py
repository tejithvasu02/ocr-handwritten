"""
Hybrid Decoder combining CTC and Attention for improved OCR accuracy.
Implements beam search with CTC prefix scoring and attention rescoring.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from heapq import heappush, heappop


@dataclass
class Hypothesis:
    """Beam search hypothesis."""
    sequence: List[int]
    score: float
    ctc_score: float
    attention_score: float
    text: str = ""
    
    def __lt__(self, other):
        return self.score > other.score  # Higher score = better


class CTCPrefixScorer:
    """
    CTC prefix scoring for beam search.
    Computes probability of prefix sequences.
    """
    
    def __init__(self, blank_id: int = 0):
        self.blank_id = blank_id
    
    def compute_prefix_scores(
        self,
        ctc_log_probs: np.ndarray,
        prefix: List[int]
    ) -> Tuple[float, float]:
        """
        Compute CTC prefix probability using forward algorithm.
        
        Args:
            ctc_log_probs: CTC log probabilities (T, vocab_size)
            prefix: Current prefix sequence
        
        Returns:
            Tuple of (prefix_prob, prefix_prob_with_blank)
        """
        T, V = ctc_log_probs.shape
        
        if not prefix:
            # Empty prefix - just blanks
            return 0.0, np.sum(ctc_log_probs[:, self.blank_id])
        
        # Initialize forward variables
        # p_b[t]: prob of prefix ending with blank at time t
        # p_nb[t]: prob of prefix ending with non-blank at time t
        p_b = np.full(T + 1, -np.inf)
        p_nb = np.full(T + 1, -np.inf)
        p_b[0] = 0.0
        
        for t in range(T):
            for s, c in enumerate(prefix):
                if s == 0:
                    # First character
                    p_nb[s + 1] = np.logaddexp(
                        p_nb[s + 1],
                        p_b[s] + ctc_log_probs[t, c]
                    )
                else:
                    # Subsequent characters
                    if c == prefix[s - 1]:
                        # Same as previous - need blank between
                        p_nb[s + 1] = np.logaddexp(
                            p_nb[s + 1],
                            p_b[s] + ctc_log_probs[t, c]
                        )
                    else:
                        p_nb[s + 1] = np.logaddexp(
                            p_nb[s + 1],
                            np.logaddexp(p_b[s], p_nb[s]) + ctc_log_probs[t, c]
                        )
                
                # Blank transition
                p_b[s + 1] = np.logaddexp(
                    p_b[s + 1],
                    np.logaddexp(p_b[s], p_nb[s]) + ctc_log_probs[t, self.blank_id]
                )
        
        n = len(prefix)
        return np.logaddexp(p_b[n], p_nb[n]), p_b[n]


class HybridDecoder:
    """
    Hybrid CTC-Attention decoder for improved OCR accuracy.
    Combines CTC prefix scoring with attention-based decoding.
    """
    
    def __init__(
        self,
        tokenizer,
        ctc_weight: float = 0.3,
        beam_size: int = 5,
        max_length: int = 128,
        length_penalty: float = 1.0,
        blank_id: int = 0
    ):
        """
        Args:
            tokenizer: Tokenizer for decoding
            ctc_weight: Weight for CTC score (0-1)
            beam_size: Number of beams for beam search
            max_length: Maximum sequence length
            length_penalty: Length normalization penalty
            blank_id: CTC blank token ID
        """
        self.tokenizer = tokenizer
        self.ctc_weight = ctc_weight
        self.attention_weight = 1.0 - ctc_weight
        self.beam_size = beam_size
        self.max_length = max_length
        self.length_penalty = length_penalty
        self.ctc_scorer = CTCPrefixScorer(blank_id)
    
    def decode(
        self,
        encoder_output: np.ndarray,
        ctc_log_probs: Optional[np.ndarray] = None,
        attention_model_fn=None
    ) -> List[Hypothesis]:
        """
        Perform hybrid decoding with beam search.
        
        Args:
            encoder_output: Encoder hidden states
            ctc_log_probs: CTC output probabilities (optional)
            attention_model_fn: Function to compute attention scores
        
        Returns:
            List of Hypothesis sorted by score
        """
        # Initialize beam
        initial_hyp = Hypothesis(
            sequence=[self.tokenizer.bos_token_id or 0],
            score=0.0,
            ctc_score=0.0,
            attention_score=0.0
        )
        
        beams = [initial_hyp]
        completed = []
        
        for step in range(self.max_length):
            candidates = []
            
            for hyp in beams:
                if self._is_complete(hyp):
                    completed.append(hyp)
                    continue
                
                # Get next token probabilities from attention
                if attention_model_fn:
                    next_log_probs = attention_model_fn(
                        encoder_output, hyp.sequence
                    )
                else:
                    # Fallback: uniform distribution
                    vocab_size = self.tokenizer.vocab_size
                    next_log_probs = np.log(np.ones(vocab_size) / vocab_size)
                
                # Get top-k candidates
                top_k_ids = np.argsort(next_log_probs)[-self.beam_size * 2:][::-1]
                
                for token_id in top_k_ids:
                    new_sequence = hyp.sequence + [token_id]
                    
                    # Attention score
                    attention_score = hyp.attention_score + next_log_probs[token_id]
                    
                    # CTC score (if available)
                    ctc_score = hyp.ctc_score
                    if ctc_log_probs is not None:
                        prefix_score, _ = self.ctc_scorer.compute_prefix_scores(
                            ctc_log_probs, new_sequence[1:]  # Skip BOS
                        )
                        ctc_score = prefix_score
                    
                    # Combined score
                    combined_score = (
                        self.attention_weight * attention_score +
                        self.ctc_weight * ctc_score
                    )
                    
                    # Length normalization
                    normalized_score = combined_score / (len(new_sequence) ** self.length_penalty)
                    
                    new_hyp = Hypothesis(
                        sequence=new_sequence,
                        score=normalized_score,
                        ctc_score=ctc_score,
                        attention_score=attention_score
                    )
                    
                    candidates.append(new_hyp)
            
            # Prune to beam size
            candidates.sort(key=lambda h: h.score, reverse=True)
            beams = candidates[:self.beam_size]
            
            if not beams:
                break
        
        # Add remaining beams to completed
        completed.extend(beams)
        
        # Decode all hypotheses
        for hyp in completed:
            hyp.text = self.tokenizer.decode(
                hyp.sequence,
                skip_special_tokens=True
            )
        
        # Sort by score
        completed.sort(key=lambda h: h.score, reverse=True)
        
        return completed[:self.beam_size]
    
    def _is_complete(self, hyp: Hypothesis) -> bool:
        """Check if hypothesis is complete (ends with EOS)."""
        eos_id = self.tokenizer.eos_token_id
        if eos_id is not None and hyp.sequence and hyp.sequence[-1] == eos_id:
            return True
        return len(hyp.sequence) >= self.max_length


class MultiPassDecoder:
    """
    Multi-pass OCR decoder for best hypothesis selection.
    Runs multiple decoding passes with different parameters.
    """
    
    def __init__(
        self,
        base_decoder: Optional[HybridDecoder] = None,
        num_passes: int = 3,
        temperature_range: Tuple[float, float] = (0.7, 1.3)
    ):
        """
        Args:
            base_decoder: Base decoder to use
            num_passes: Number of decoding passes
            temperature_range: Range of temperatures to sample
        """
        self.base_decoder = base_decoder
        self.num_passes = num_passes
        self.temperature_range = temperature_range
    
    def decode_multi_pass(
        self,
        encoder_output: np.ndarray,
        ctc_log_probs: Optional[np.ndarray] = None,
        attention_model_fn=None,
        scoring_fn=None
    ) -> Tuple[str, float, List[Dict]]:
        """
        Perform multi-pass decoding and select best hypothesis.
        
        Args:
            encoder_output: Encoder hidden states
            ctc_log_probs: CTC output probabilities
            attention_model_fn: Attention score function
            scoring_fn: Optional function to score hypotheses
        
        Returns:
            Tuple of (best_text, confidence, all_hypotheses)
        """
        all_hypotheses = []
        
        for pass_idx in range(self.num_passes):
            # Vary temperature for diversity
            temp = np.linspace(
                self.temperature_range[0],
                self.temperature_range[1],
                self.num_passes
            )[pass_idx]
            
            # Apply temperature to CTC probs if available
            adjusted_ctc = None
            if ctc_log_probs is not None:
                adjusted_ctc = ctc_log_probs / temp
            
            # Decode
            if self.base_decoder:
                hypotheses = self.base_decoder.decode(
                    encoder_output,
                    adjusted_ctc,
                    attention_model_fn
                )
                
                for hyp in hypotheses:
                    all_hypotheses.append({
                        'text': hyp.text,
                        'score': hyp.score,
                        'pass': pass_idx,
                        'temperature': temp
                    })
        
        if not all_hypotheses:
            return "", 0.0, []
        
        # Score and select best
        if scoring_fn:
            for hyp in all_hypotheses:
                hyp['external_score'] = scoring_fn(hyp['text'])
                hyp['combined_score'] = (hyp['score'] + hyp['external_score']) / 2
        else:
            for hyp in all_hypotheses:
                hyp['combined_score'] = hyp['score']
        
        # Sort by combined score
        all_hypotheses.sort(key=lambda h: h['combined_score'], reverse=True)
        
        best = all_hypotheses[0]
        
        # Confidence based on agreement
        unique_texts = set(h['text'] for h in all_hypotheses[:3])
        confidence = 1.0 / len(unique_texts)  # Higher if texts agree
        
        return best['text'], confidence, all_hypotheses


class ConfidenceScorer:
    """
    Sequence-level confidence scoring for OCR output.
    """
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        use_entropy: bool = True
    ):
        self.min_confidence = min_confidence
        self.use_entropy = use_entropy
    
    def compute_confidence(
        self,
        token_probs: np.ndarray,
        decoded_text: str
    ) -> Tuple[float, Dict]:
        """
        Compute sequence-level confidence score.
        
        Args:
            token_probs: Per-token probabilities
            decoded_text: Decoded text string
        
        Returns:
            Tuple of (confidence, details)
        """
        if len(token_probs) == 0:
            return 0.0, {'reason': 'empty_sequence'}
        
        # Mean probability
        mean_prob = np.mean(token_probs)
        
        # Min probability (weak link)
        min_prob = np.min(token_probs)
        
        # Entropy-based measure
        entropy = 0.0
        if self.use_entropy:
            # Lower entropy = higher confidence
            entropy = -np.mean(token_probs * np.log(token_probs + 1e-10))
            entropy_score = 1.0 / (1.0 + entropy)
        else:
            entropy_score = 1.0
        
        # Combined confidence
        confidence = (
            0.4 * mean_prob +
            0.3 * min_prob +
            0.3 * entropy_score
        )
        
        details = {
            'mean_prob': float(mean_prob),
            'min_prob': float(min_prob),
            'entropy': float(entropy),
            'entropy_score': float(entropy_score)
        }
        
        return float(confidence), details
    
    def should_reject(
        self,
        confidence: float,
        text: str
    ) -> Tuple[bool, str]:
        """
        Determine if output should be rejected.
        
        Returns:
            Tuple of (should_reject, reason)
        """
        if confidence < self.min_confidence:
            return True, f"confidence_below_threshold ({confidence:.2f} < {self.min_confidence})"
        
        if not text.strip():
            return True, "empty_output"
        
        # Check for nonsensical patterns
        if len(set(text)) < 3 and len(text) > 5:
            return True, "low_character_diversity"
        
        return False, "passed"
