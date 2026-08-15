from engine.helper import DEBUG


class Runtime:
    def load(self, model_path): print(f'Loading model from {model_path}')
    def generate(self, prompt):
       if DEBUG>1: print(f"Prompt: {prompt}")
       return "Here should be the model answer..." 