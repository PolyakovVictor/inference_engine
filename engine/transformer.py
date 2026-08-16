from typing import Any
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
from tinygrad.nn.state import get_state_dict, safe_save, load_state_dict, safe_load

from .config import ModelConfig


class SelfAttention:
    """
    Механізм Multi-Head Attention з каузальною (causal) маскою для авторегресійних моделей.
    """
    def __init__(self, dim: int, n_heads: int) -> None:
        # Розмірність моделі має націло ділитися на кількість голів
        assert dim % n_heads == 0

        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        # Лінійні проєкції для векторів Query, Key, Value та фінального виходу
        self.q_proj = Linear(dim, dim)
        self.k_proj = Linear(dim, dim)
        self.v_proj = Linear(dim, dim)
        self.o_proj = Linear(dim, dim)
        
    def __call__(self, x: Tensor) -> Any:
        # x shape: [batch, sequence_length, dimension]
        batch, seq_len, _ = x.shape

        # 1. Проєктуємо вхідний тензор у Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # 2. Розбиваємо простір ознак на незалежні голови уваги (Multi-Head)
        # [batch, seq, dim] -> [batch, seq, n_heads, head_dim]
        q = q.reshape(batch, seq_len, self.n_heads, self.head_dim)
        k = k.reshape(batch, seq_len, self.n_heads, self.head_dim)
        v = v.reshape(batch, seq_len, self.n_heads, self.head_dim)

        # 3. Змінюємо осі для паралельного матричного множення по кожній голові
        # [batch, seq, n_heads, head_dim] -> [batch, n_heads, seq, head_dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # 4. Обчислюємо ваги уваги (скалярний добуток Q і транспонованого K)
        # scores shape: [batch, n_heads, seq, seq]
        scores: Tensor = q.matmul(k.transpose(2, 3))
        
        # Масштабування (Scaled Dot-Product) для запобігання надто малим градієнтам при Softmax
        scores = scores / (self.head_dim ** 0.5)

        # 5. Каузальна маска (трикутна матриця): забороняє токенам дивитися в майбутнє
        mask = Tensor.ones(seq_len, seq_len).tril()
        mask = mask.reshape(1, 1, seq_len, seq_len)

        # Заповнюємо майбутні позиції -inf, щоб softmax перетворив їх на 0
        scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = scores.softmax(axis=-1)

        # 6. Зважене підсумовування значень (Values)
        # output shape: [batch, n_heads, seq, head_dim]
        output = attention.matmul(v)

        # 7. Збираємо всі голови назад в єдиний вихідний вектор розмірності dim
        # [batch, n_heads, seq, head_dim] -> [batch, seq, n_heads, head_dim] -> [batch, seq, dim]
        output = output.transpose(1, 2)
        output = output.reshape(batch, seq_len, self.dim)
        
        # Фінальна вихідна лінійна трансформація
        return self.o_proj(output)


class MLP:
    """
    Повнозв'язна мережа (Feed-Forward Network), що обробляє кожен токен незалежно.
    """
    def __init__(self, dim: int, hidden_dim: int) -> None:
        # Збільшуємо розмірність до hidden_dim для нелінійної сепарації
        self.fc1 = Linear(dim, hidden_dim)
        # Повертаємо назад до базової розмірності dim
        self.fc2 = Linear(hidden_dim, dim)

    def __call__(self, x: Tensor) -> Any:
        x = self.fc1(x)
        x = x.gelu()  # Нелінійна функція активації GELU
        x = self.fc2(x)
        return x


class TransformerBlock:
    """
    Базовий блок трансформера: з'єднує Attention і MLP через Residual Connections.
    """
    def __init__(self, dim: int, n_heads: int, hidden_dim: int) -> None:
        self.attention = SelfAttention(dim, n_heads)
        self.mlp = MLP(dim, hidden_dim)
    
    def __call__(self, x: Tensor) -> Any:
        # Residual connection 1: додаємо вхід до виходу Attention
        x = x + self.attention(x)
        # Residual connection 2: додаємо результат до виходу MLP
        x = x + self.mlp(x)
        return x


class TinyLLM:
    """
    Головний клас декодерної мовної моделі (GPT-подібна архітектура).
    """
    def __init__(self, config: ModelConfig) -> None:
        self.config = config
        self.vocab_size = config.vocab_size
        self.dim = config.dim

        # Таблиця ембедингів для токенів [vocab_size, dim]
        self.token_embedding = Tensor.uniform(config.vocab_size, config.dim, low=-0.1, high=0.1)
        # Таблиця позиційних ембедингів (фіксований контекст до 1024 токенів)
        self.position_embedding = Tensor.uniform(1024, config.dim, low=-0.1, high=0.1)
        
        # Стек послідовних блоків трансформера
        self.blocks = [
            TransformerBlock(config.dim, config.n_heads, config.hidden_dim) 
            for _ in range(config.n_layers)
        ]
        
        # Проєкція виходу моделі назад у простір словника (отримання logits)
        self.lm_head = Linear(config.dim, config.vocab_size)
 
    def __call__(self, tokens: Tensor) -> Tensor:
        # tokens shape: [batch, seq_len]
        seq_len = tokens.shape[-1]
        
        # Отримуємо векторні представлення слів за їхніми ID
        x = self.token_embedding[tokens]
        
        # Генеруємо індекси позицій [0, 1, ..., seq_len - 1] та додаємо позиційні ембединги
        positions = Tensor.arange(seq_len)
        x = x + self.position_embedding[positions]
        
        # Послідовний прогін через усі трансформерні шари
        for block in self.blocks: 
            x = block(x)
            
        # Обчислюємо нескориговані ймовірності для кожного слова у словнику
        logits = self.lm_head(x)
        return logits
    
    def generate(self, tokens: Tensor, max_new_tokens: int = 20) -> Tensor:
        """
        Жадібна генерація (Greedy Search): на кожному кроці обирається токен з найбільшим logit.
        """
        for _ in range(max_new_tokens):
            # Прогін поточної послідовності через модель
            logits = self(tokens)
            
            # Беремо передбачення виключно для останнього токена в послідовності
            next_logits = logits[:, -1, :]
            
            # Жадібний вибір токена з максимальною ймовірністю
            next_tokens = next_logits.argmax(axis=-1)
            
            # Додаємо новий токен у кінець послідовності для наступної ітерації
            tokens = tokens.cat(next_tokens.unsqueeze(1), dim=1)
            
        return tokens
    
    def save(self, path: str): safe_save(get_state_dict(self), path)
    def load(self, path: str): load_state_dict(self, safe_load(path))