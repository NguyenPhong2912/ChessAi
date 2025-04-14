import os
import sys
import numpy as np
import torch
from datetime import datetime
import json
from typing import List, Dict, Any
from collections import deque

# Thêm đường dẫn gốc vào Python path
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
except NameError:
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))

sys.path.insert(0, root_dir)

from chess_env import ChessEnv
from dqn_agent import DQNAgent
from meta_controller import MetaControllerTrainer

class AdaptiveTrainer:
    def __init__(self, num_episodes: int = 1000, eval_interval: int = 50,
                 window_size: int = 100, target_performance: float = 0.6):
        """
        Khởi tạo Adaptive Trainer
        
        Args:
            num_episodes (int): Số episode huấn luyện
            eval_interval (int): Tần suất đánh giá
            window_size (int): Kích thước cửa sổ cho metrics
            target_performance (float): Hiệu suất mục tiêu
        """
        self.num_episodes = num_episodes
        self.eval_interval = eval_interval
        self.window_size = window_size
        self.target_performance = target_performance
        
        # Khởi tạo môi trường
        self.env = ChessEnv()
        
        # Khởi tạo DQN agent
        self.agent = DQNAgent(
            state_channels=12,  # 6 loại quân * 2 màu
            board_size=8,
            action_size=64 * 64,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            memory_size=10000,
            batch_size=64,
            target_update=1000
        )
        
        # Khởi tạo meta-controller
        self.meta_controller = MetaControllerTrainer(
            input_size=10,  # Số metrics
            hidden_size=64,
            learning_rate=0.0001
        )
        
        # Lưu trữ metrics
        self.rewards_history = deque(maxlen=window_size)
        self.losses_history = deque(maxlen=window_size)
        self.epsilon_history = deque(maxlen=window_size)
        
        # Tạo thư mục lưu model và logs
        self.model_dir = os.path.join(root_dir, 'models', 'adaptive')
        self.logs_dir = os.path.join(root_dir, 'logs', 'adaptive')
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
    def get_metrics(self) -> List[float]:
        """
        Lấy vector metrics từ lịch sử
        
        Returns:
            List[float]: Vector metrics
        """
        if len(self.rewards_history) < self.window_size:
            return [0.0] * 10
            
        return [
            np.mean(self.rewards_history),
            np.std(self.rewards_history),
            np.mean(self.losses_history),
            np.std(self.losses_history),
            np.mean(self.epsilon_history),
            np.std(self.epsilon_history),
            self.agent.epsilon,
            self.agent.steps,
            len(self.agent.memory),
            self.agent.optimizer.param_groups[0]['lr']
        ]
        
    def train_episode(self) -> Dict[str, float]:
        """
        Huấn luyện một episode
        
        Returns:
            Dict[str, float]: Thống kê của episode
        """
        state = self.env.reset()
        total_reward = 0
        total_loss = 0
        moves = 0
        done = False
        
        while not done and moves < 100:  # Giới hạn số nước đi
            # Lấy danh sách nước đi hợp lệ
            legal_moves = self.env.engine.get_legal_moves()
            
            # Chọn hành động
            action = self.agent.act(state, legal_moves)
            
            # Thực hiện hành động
            next_state, reward, done = self.env.step(action)
            
            # Lưu transition
            self.agent.remember(state, action, reward, next_state, done)
            
            # Học từ replay buffer
            loss = self.agent.replay()
            
            # Cập nhật metrics
            total_reward += reward
            total_loss += loss
            moves += 1
            
            state = next_state
            
        return {
            'reward': total_reward,
            'loss': total_loss / moves if moves > 0 else 0,
            'moves': moves
        }
        
    def evaluate(self) -> float:
        """
        Đánh giá hiệu suất của agent
        
        Returns:
            float: Tỷ lệ thắng
        """
        wins = 0
        for _ in range(10):  # Chơi 10 ván đánh giá
            state = self.env.reset()
            done = False
            moves = 0
            
            while not done and moves < 100:
                legal_moves = self.env.engine.get_legal_moves()
                action = self.agent.act(state, legal_moves)
                state, _, done = self.env.step(action)
                moves += 1
                
            if self.env.engine.is_checkmate():
                wins += 1
                
        return wins / 10
        
    def train(self):
        """Thực hiện quá trình huấn luyện"""
        best_win_rate = 0
        best_model_path = os.path.join(self.model_dir, 'best_model.pth')
        best_meta_path = os.path.join(self.model_dir, 'best_meta.pth')
        
        for episode in range(self.num_episodes):
            # Huấn luyện episode
            stats = self.train_episode()
            
            # Cập nhật metrics
            self.rewards_history.append(stats['reward'])
            self.losses_history.append(stats['loss'])
            self.epsilon_history.append(self.agent.epsilon)
            
            # Đánh giá định kỳ
            if (episode + 1) % self.eval_interval == 0:
                win_rate = self.evaluate()
                print(f"\nEpisode {episode + 1}/{self.num_episodes}")
                print(f"Win rate: {win_rate:.2%}")
                print(f"Average reward: {np.mean(self.rewards_history):.2f}")
                print(f"Average loss: {np.mean(self.losses_history):.4f}")
                print(f"Epsilon: {self.agent.epsilon:.4f}")
                
                # Lưu model tốt nhất
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    self.agent.save(best_model_path)
                    self.meta_controller.save(best_meta_path)
                    print(f"New best model saved with win rate: {win_rate:.2%}")
                
                # Cập nhật meta-controller
                metrics = self.get_metrics()
                factors = self.meta_controller.update(metrics, self.target_performance)
                
                # Áp dụng các hệ số điều chỉnh
                self.agent.epsilon_decay *= factors[0]
                self.agent.optimizer.param_groups[0]['lr'] *= factors[1]
                self.agent.gamma *= factors[2]
                
                # Lưu log
                self.save_log(episode, stats, win_rate, factors)
                
    def save_log(self, episode: int, stats: Dict[str, float],
                win_rate: float, factors: tuple):
        """
        Lưu log huấn luyện
        
        Args:
            episode (int): Số episode
            stats (Dict[str, float]): Thống kê episode
            win_rate (float): Tỷ lệ thắng
            factors (tuple): Các hệ số điều chỉnh
        """
        log_data = {
            'episode': episode,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stats': stats,
            'win_rate': win_rate,
            'metrics': {
                'avg_reward': np.mean(self.rewards_history),
                'avg_loss': np.mean(self.losses_history),
                'epsilon': self.agent.epsilon,
                'memory_size': len(self.agent.memory)
            },
            'adjustment_factors': {
                'epsilon_decay': factors[0],
                'learning_rate': factors[1],
                'gamma': factors[2]
            }
        }
        
        log_path = os.path.join(self.logs_dir, f'train_{datetime.now().strftime("%d_%m_%Y_%H")}.log')
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=4)

def main():
    """Hàm chính"""
    trainer = AdaptiveTrainer(
        num_episodes=1000,
        eval_interval=50,
        window_size=100,
        target_performance=0.6
    )
    
    print("Bắt đầu huấn luyện với Adaptive Meta-Controller...")
    trainer.train()
    print("Hoàn thành huấn luyện!")

if __name__ == "__main__":
    main() 