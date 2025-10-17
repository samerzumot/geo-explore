"""LLM client for geological data extraction."""

import json
import logging
from typing import Dict, Any, List, Optional, Union
import asyncio
from pathlib import Path

import ollama
import openai
from openai import AsyncOpenAI

from geoextract.config import settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Client for interacting with various LLM providers."""
    
    def __init__(self, provider: str = None, model: str = None):
        """Initialize LLM client.
        
        Args:
            provider: LLM provider ('ollama', 'openai', 'anthropic')
            model: Model name to use
        """
        self.provider = provider or settings.llm_provider
        self.model = model or settings.llm_model
        
        # Initialize clients
        self.ollama_client = None
        self.openai_client = None
        
        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            # Test connection
            models = ollama.list()
            available_models = [model['name'] for model in models['models']]
            
            if self.model not in available_models:
                logger.warning(f"Model {self.model} not found. Available: {available_models}")
                # Use first available model as fallback
                if available_models:
                    self.model = available_models[0]
                    logger.info(f"Using fallback model: {self.model}")
                else:
                    raise RuntimeError("No Ollama models available")
            
            self.ollama_client = ollama
            logger.info(f"Ollama client initialized with model: {self.model}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama: {e}")
            raise
    
    def _init_openai(self):
        """Initialize OpenAI client."""
        if not settings.openai_api_key:
            raise ValueError("OpenAI API key not configured")
        
        try:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
            logger.info(f"OpenAI client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI: {e}")
            raise
    
    def _init_anthropic(self):
        """Initialize Anthropic client."""
        if not settings.anthropic_api_key:
            raise ValueError("Anthropic API key not configured")
        
        try:
            import anthropic
            self.anthropic_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
            logger.info(f"Anthropic client initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"Failed to initialize Anthropic: {e}")
            raise
    
    async def extract_entities(self, text: str, prompt: str) -> Dict[str, Any]:
        """Extract entities from text using LLM.
        
        Args:
            text: Input text to process
            prompt: System prompt for extraction
            
        Returns:
            Dictionary with extracted entities
        """
        try:
            if self.provider == "ollama":
                return await self._extract_with_ollama(text, prompt)
            elif self.provider == "openai":
                return await self._extract_with_openai(text, prompt)
            elif self.provider == "anthropic":
                return await self._extract_with_anthropic(text, prompt)
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except Exception as e:
            logger.error(f"Entity extraction failed: {e}")
            return {
                "error": str(e),
                "entities": {},
                "confidence": 0.0
            }
    
    async def _extract_with_ollama(self, text: str, prompt: str) -> Dict[str, Any]:
        """Extract entities using Ollama.
        
        Args:
            text: Input text
            prompt: System prompt
            
        Returns:
            Extraction results
        """
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract geological data from this text:\n\n{text}"}
            ]
            
            # Make request
            response = self.ollama_client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.1,  # Low temperature for consistent extraction
                    "top_p": 0.9,
                }
            )
            
            # Parse response
            content = response['message']['content']
            
            # Try to parse as JSON
            try:
                entities = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    entities = json.loads(json_match.group())
                else:
                    # Fallback: return raw content
                    entities = {"raw_content": content}
            
            return {
                "entities": entities,
                "confidence": 0.8,  # Ollama doesn't provide confidence scores
                "model": self.model,
                "provider": "ollama"
            }
            
        except Exception as e:
            logger.error(f"Ollama extraction failed: {e}")
            raise
    
    async def _extract_with_openai(self, text: str, prompt: str) -> Dict[str, Any]:
        """Extract entities using OpenAI.
        
        Args:
            text: Input text
            prompt: System prompt
            
        Returns:
            Extraction results
        """
        try:
            # Prepare messages
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract geological data from this text:\n\n{text}"}
            ]
            
            # Make request
            response = await self.openai_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.1,
                top_p=0.9,
                response_format={"type": "json_object"}
            )
            
            # Parse response
            content = response.choices[0].message.content
            entities = json.loads(content)
            
            return {
                "entities": entities,
                "confidence": 0.9,  # OpenAI generally provides high quality
                "model": self.model,
                "provider": "openai",
                "usage": {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            raise
    
    async def _extract_with_anthropic(self, text: str, prompt: str) -> Dict[str, Any]:
        """Extract entities using Anthropic.
        
        Args:
            text: Input text
            prompt: System prompt
            
        Returns:
            Extraction results
        """
        try:
            # Prepare message
            message = f"{prompt}\n\nExtract geological data from this text:\n\n{text}"
            
            # Make request
            response = await self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=4000,
                temperature=0.1,
                messages=[{"role": "user", "content": message}]
            )
            
            # Parse response
            content = response.content[0].text
            
            # Try to parse as JSON
            try:
                entities = json.loads(content)
            except json.JSONDecodeError:
                # If not JSON, try to extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    entities = json.loads(json_match.group())
                else:
                    # Fallback: return raw content
                    entities = {"raw_content": content}
            
            return {
                "entities": entities,
                "confidence": 0.9,  # Anthropic generally provides high quality
                "model": self.model,
                "provider": "anthropic",
                "usage": {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Anthropic extraction failed: {e}")
            raise
    
    def extract_entities_sync(self, text: str, prompt: str) -> Dict[str, Any]:
        """Synchronous version of extract_entities.
        
        Args:
            text: Input text to process
            prompt: System prompt for extraction
            
        Returns:
            Dictionary with extracted entities
        """
        return asyncio.run(self.extract_entities(text, prompt))
    
    async def batch_extract(self, texts: List[str], prompt: str) -> List[Dict[str, Any]]:
        """Extract entities from multiple texts.
        
        Args:
            texts: List of input texts
            prompt: System prompt for extraction
            
        Returns:
            List of extraction results
        """
        tasks = [self.extract_entities(text, prompt) for text in texts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch extraction failed for text {i}: {result}")
                processed_results.append({
                    "error": str(result),
                    "entities": {},
                    "confidence": 0.0
                })
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model.
        
        Returns:
            Dictionary with model information
        """
        return {
            "provider": self.provider,
            "model": self.model,
            "status": "available" if self._is_available() else "unavailable"
        }
    
    def _is_available(self) -> bool:
        """Check if the LLM client is available.
        
        Returns:
            True if available
        """
        try:
            if self.provider == "ollama" and self.ollama_client:
                return True
            elif self.provider == "openai" and self.openai_client:
                return True
            elif self.provider == "anthropic" and hasattr(self, 'anthropic_client'):
                return True
            return False
        except Exception:
            return False
    
    def test_connection(self) -> Dict[str, Any]:
        """Test connection to LLM provider.
        
        Returns:
            Dictionary with test results
        """
        try:
            test_text = "Test geological data: 45.1234°N, 123.4567°W, Gold assay: 2.5 g/t"
            test_prompt = "Extract coordinates and assay data as JSON."
            
            result = self.extract_entities_sync(test_text, test_prompt)
            
            return {
                "status": "success",
                "provider": self.provider,
                "model": self.model,
                "test_result": result
            }
        except Exception as e:
            return {
                "status": "failed",
                "provider": self.provider,
                "model": self.model,
                "error": str(e)
            }