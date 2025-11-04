"""
Pydantic models for API request/response validation
"""

from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    """Request model for text generation endpoint."""
    
    seed_text: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Starting text for generation"
    )
    num_words: int = Field(
        50,
        ge=1,
        le=500,
        description="Number of words to generate"
    )
    temperature: float = Field(
        0.75,
        ge=0.1,
        le=2.0,
        description="Sampling temperature (0.5-0.7=coherent, 0.8-1.0=balanced, >1.0=creative)"
    )
    top_k: int = Field(
        50,
        ge=0,
        le=200,
        description="Keep only top-k most likely tokens. Higher=more diverse (50-80 recommended)"
    )
    top_p: float = Field(
        0.92,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold (0.85-0.95 recommended for coherence)"
    )
    use_beam_search: bool = Field(
        False,
        description="Whether to use beam search for better quality (slower but more coherent)"
    )
    beam_width: int = Field(
        3,
        ge=1,
        le=10,
        description="Number of beams for beam search (3-5 recommended)"
    )
    
    class Config:
        example = {
            "seed_text": "the cat sat on",
            "num_words": 20,
            "temperature": 0.8,
            "top_k": 40,
            "top_p": 0.9,
            "use_beam_search": True,
            "beam_width": 3
        }


class GenerateResponse(BaseModel):
    """Response model for text generation endpoint."""
    
    seed_text: str = Field(description="The seed text provided")
    generated_text: str = Field(description="Complete generated text (seed + new)")
    num_words_generated: int = Field(description="Number of words actually generated")
    temperature: float = Field(description="Temperature used for generation")
    
    class Config:
        example = {
            "seed_text": "the cat sat on",
            "generated_text": "the cat sat on the mat and slept peacefully...",
            "num_words_generated": 20,
            "temperature": 1.0
        }


class ModelInfo(BaseModel):
    """Model information response."""
    
    vocabulary_size: int = Field(description="Total vocabulary size")
    sequence_length: int = Field(description="Sequence length used in training")
    embedding_dim: int = Field(description="Embedding dimension")
    lstm_units: int = Field(description="Number of units in LSTM layers")
    num_layers: int = Field(description="Number of LSTM layers")
    is_loaded: bool = Field(description="Whether model is currently loaded")
    
    class Config:
        example = {
            "vocabulary_size": 10000,
            "sequence_length": 50,
            "embedding_dim": 100,
            "lstm_units": 150,
            "num_layers": 2,
            "is_loaded": True
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(description="Health status")
    model_loaded: bool = Field(description="Whether model is loaded")
    
    class Config:
        example = {
            "status": "healthy",
            "model_loaded": True
        }


class ErrorResponse(BaseModel):
    """Error response."""
    
    error: str = Field(description="Error message")
    detail: Optional[str] = Field(None, description="Additional error details")
