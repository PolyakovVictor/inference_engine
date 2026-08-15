from pathlib import Path


class ModelManager:
    def __init__(self, models_dir="models") -> None:
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def get_model_path(self, name: str):
        return self.models_dir / name