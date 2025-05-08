import os
import requests
from typing import List, Dict, Any
from dotenv import load_dotenv

class SimpleContainerPredictor:
    """
    A simple class that predicts what might be inside containers based on scene context.
    """
    def __init__(self, api_key=None):
        """
        Initialize the predictor.
        
        Args:
            api_key: Optional API key for Claude. If not provided, will look for ANTHROPIC_API_KEY in environment.
        """
        # Load API key from environment if not provided
        if api_key is None:
            load_dotenv()
            api_key = 
            if api_key is None:
                raise ValueError("No API key provided and ANTHROPIC_API_KEY not found in environment")
        
        self.api_key = api_key
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.model = "claude-3-haiku-20240307"
        
        # Cache for previous predictions
        self.prediction_cache = {}
    
    def predict_container_contents(self, container_type: str, objects_in_scene: List[str]) -> List[str]:
        """
        Predict what might be inside a container based on other objects in the scene.
        
        Args:
            container_type: Type of container (e.g., "cabinet", "drawer", "box")
            objects_in_scene: List of other objects visible in the scene
            
        Returns:
            List of predicted items inside the container
        """
        # Create a cache key
        cache_key = f"{container_type}:{','.join(sorted(objects_in_scene))}"
        
        # Check cache first
        if cache_key in self.prediction_cache:
            return self.prediction_cache[cache_key]
        
        # Format objects into a comma-separated string
        objects_str = ", ".join(objects_in_scene)
        
        # Create the prompt
        prompt = f"In a space containing {objects_str}, what items would you typically find inside a {container_type}? Return only a simple comma-separated list of items, no explanations."
        
        # Query the model
        try:
            items = self._query_claude(prompt)
            
            # Clean up items
            cleaned_items = [item.strip().lower() for item in items.split(",")]
            
            # Store in cache
            self.prediction_cache[cache_key] = cleaned_items
            
            return cleaned_items
            
        except Exception as e:
            print(f"Error querying model: {e}")
            # Return basic fallback predictions
            return self._get_fallback_predictions(container_type)
    
    def _query_claude(self, prompt: str) -> str:
        """
        Query Claude with a prompt.
        
        Args:
            prompt: The prompt to send to Claude
            
        Returns:
            Claude's response text
        """
        headers = {
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
            "x-api-key": self.api_key
        }
        
        data = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
            "max_tokens": 300
        }
        
        response = requests.post(self.api_url, headers=headers, json=data)
        response.raise_for_status()
        
        return response.json()["content"][0]["text"].strip()
    
    def _get_fallback_predictions(self, container_type: str) -> List[str]:
        """
        Get fallback predictions if the model query fails.
        
        Args:
            container_type: Type of container
            
        Returns:
            List of fallback predictions
        """
        fallbacks = {
            "cabinet": ["plates", "cups", "glasses", "bowls"],
            "drawer": ["utensils", "tools", "pens", "papers"],
            "refrigerator": ["food", "drinks", "vegetables", "fruits"],
            "box": ["items", "tools", "supplies"],
            "desk": ["papers", "pens", "laptop", "notebook"],
            "closet": ["clothes", "shoes", "hangers"],
            "bookshelf": ["books", "magazines", "decor"],
        }
        
        return fallbacks.get(container_type.lower(), ["items"])


# Simple example usage
if __name__ == "__main__":
    predictor = SimpleContainerPredictor()
    
    # Test kitchen scene
    kitchen_items = ["stove", "refrigerator", "microwave", "sink"]
    cabinet_predictions = predictor.predict_container_contents("cabinet", kitchen_items)
    print(f"In a kitchen with {', '.join(kitchen_items)}, a cabinet might contain: {', '.join(cabinet_predictions)}")
    
    # Test office scene
    office_items = ["desk", "chair", "computer", "printer"]
    drawer_predictions = predictor.predict_container_contents("drawer", office_items)
    print(f"In an office with {', '.join(office_items)}, a drawer might contain: {', '.join(drawer_predictions)}")