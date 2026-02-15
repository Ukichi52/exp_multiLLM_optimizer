"""Target model (model to be attacked)"""
from typing import Optional
from .base_model import BaseModel

class TargetModel(BaseModel):
    """Target model wrapper"""
    
    def call(self, image_path: str, query: str) -> str:
        """
        Call target model with image and query
        
        Args:
            image_path: Path to image file
            query: Text query
            
        Returns:
            Model response text
        """
        image_base64 = self._encode_image(image_path)
        
        prefix = "Describe the image content. "
        full_text = f"{prefix}{query}"
        
        messages = [
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/jpeg;base64,{image_base64}'
                        }
                    },
                    {
                        'type': 'text',
                        'text': full_text
                    }
                ]
            }
        ]
        
        return self._call_openai_format(messages)
