"""Abstract base class for all models"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import requests
import json
import base64

class BaseModel(ABC):
    """Base class for all model wrappers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_base = config.get('api_base')
        self.api_key = config.get('api_key')
        self.model_name = config.get('model_name')
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        with open(image_path, 'rb') as f:
            return base64.b64encode(f.read()).decode('utf-8')
    
    def _call_openai_format(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Call OpenAI-format API"""
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.api_key}'
        }
        
        data = {
            'model': self.model_name,
            'messages': messages,
            'temperature': temperature,
            'max_tokens': max_tokens
        }
        
        response = requests.post(
            f'{self.api_base}/chat/completions',
            headers=headers,
            json=data
        )
        
        result = response.json()
        return result['choices'][0]['message']['content']
    
    @abstractmethod
    def call(self, **kwargs) -> str:
        """Main interface for calling the model"""
        pass