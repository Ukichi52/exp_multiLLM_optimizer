"""Abstract base class for all models"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Union, List
import requests
import json
import base64
import time

class BaseModel(ABC):
    """Base class for all model wrappers"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.api_base = config.get('api_base')
        self.api_key = config.get('api_key')
        self.model_name = config.get('model_name')
    
    def _encode_image(self, image_path: str) -> str:
        """Encode image to base64"""
        try:
            with open(image_path, 'rb') as f:
                return base64.b64encode(f.read()).decode('utf-8')
        except FileNotFoundError:
            print(f"[Error] Image not found: {image_path}")
            return ""
    
    def _call_openai_format(
        self,
        messages: list,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        retries: int = 3
    ) -> str:
        """Call OpenAI-format API with error handling and retry"""
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
        
        if self.api_base.endswith('/'):
            url = f'{self.api_base}chat/completions'
        else:
            url = f'{self.api_base}/chat/completions'

        for attempt in range(retries):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=data,
                    timeout=60 
                )
                
                if response.status_code != 200:
                    print(f"\n[API Error] Status: {response.status_code}")
                    print(f"[API Error] Response: {response.text}")
                    
                    if response.status_code in [429, 500, 502, 503, 504]:
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        return ""

                result = response.json()
                
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                elif 'error' in result:
                    print(f"\n[API Logic Error]: {result['error']}")
                    return ""
                else:
                    print(f"\n[API Format Error] Unexpected response: {result}")
                    return ""

            except requests.exceptions.RequestException as e:
                print(f"\n[Network Error] Attempt {attempt+1}/{retries}: {e}")
                time.sleep(2)
            except json.JSONDecodeError:
                print(f"\n[JSON Error] Could not decode response. Raw text: {response.text}")
                return ""
            except Exception as e:
                print(f"\n[Unknown Error] {e}")
                return ""
        
        print("[Error] Max retries reached. Returning empty string.")
        return ""
    
    @abstractmethod
    def call(self, inputs: Union[str, List[Dict[str, str]]], **kwargs) -> str:
        """Main interface for calling the model"""
        pass
