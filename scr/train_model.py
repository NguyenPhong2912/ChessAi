import os
import sys
from pathlib import Path
import torch
from datetime import datetime

# Xác định đường dẫn gốc và thêm vào Python path
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))

sys.path.insert(0, root_dir)

# Import các module từ dự án
from game.ChessEngine import ChessEngine
from agents.q_learning_agent import QLearningAgent
from training.trainer import Trainer
from training.model_evaluator import ModelEvaluator

def train_model(num_episodes=1000, max_moves=100, max_time=300):
    """Huấn luyện model cờ vua"""
    print("Bắt đầu huấn luyện model...")
    
    # Khởi tạo engine và agent
    engine = ChessEngine()
    agent = QLearningAgent(
        state_size=64 * 12,  # 64 ô, mỗi ô có 12 trạng thái
        action_size=64 * 64,  # Số lượng nước đi có thể
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        replay_capacity=10000,
        batch_size=64,
        target_update=1000
    )
    
    # Khởi tạo trainer và evaluator
    trainer = Trainer(agent, engine)
    evaluator = ModelEvaluator(agent, engine)
    
    # Tạo thư mục lưu model nếu chưa tồn tại
    model_dir = os.path.join(root_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    # Huấn luyện
    best_win_rate = 0
    for episode in range(num_episodes):
        # Huấn luyện một episode
        episode_reward, episode_loss, episode_moves = trainer.train_episode(
            max_moves=max_moves,
            max_time=max_time
        )
        
        # Đánh giá mỗi 10 episode
        if (episode + 1) % 10 == 0:
            win_rate = evaluator.evaluate(num_games=5)
            print(f"Episode {episode + 1}/{num_episodes}")
            print(f"Win rate: {win_rate:.2%}")
            print(f"Epsilon: {agent.epsilon:.4f}")
            print(f"Loss: {episode_loss:.4f}")
            print(f"Moves: {episode_moves}")
            print("-" * 50)
            
            # Lưu model tốt nhất
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                model_path = os.path.join(model_dir, 'chess_ai_final.pth')
                agent.save_model(model_path)
                print(f"Đã lưu model tốt nhất với win rate: {win_rate:.2%}")
    
    print("\nHoàn thành huấn luyện!")
    print(f"Win rate tốt nhất: {best_win_rate:.2%}")

if __name__ == "__main__":
    train_model() 