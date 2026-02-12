"""Manager for results_index.jsonl"""
import json
from pathlib import Path
from typing import Dict, List, Optional

class IndexManager:
    """Manage results index file"""
    
    def __init__(self, index_path: str):
        self.index_path = Path(index_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.index_path.exists():
            self.index_path.touch()
    
    def add_entry(self, entry: Dict):
        """Add new entry to index"""
        with open(self.index_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def load_all(self) -> List[Dict]:
        """Load all entries"""
        entries = []
        with open(self.index_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
        return entries
    
    def get_entry(self, image_id: str) -> Optional[Dict]:
        """Get specific entry by image_id"""
        for entry in self.load_all():
            if entry['image_id'] == image_id:
                return entry
        return None
    
    def update_entry(self, image_id: str, updates: Dict):
        """Update specific entry"""
        entries = self.load_all()
        
        for entry in entries:
            if entry['image_id'] == image_id:
                entry.update(updates)
                break
        
        with open(self.index_path, 'w', encoding='utf-8') as f:
            for entry in entries:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    def get_pending_items(self) -> List[Dict]:
        """Get all items with AS=0"""
        return [e for e in self.load_all() if e.get('AS') == 0]