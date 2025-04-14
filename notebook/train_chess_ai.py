import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
import time
import json

# Xác định đường dẫn gốc và thêm vào Python path
try:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
except NameError:
    root_dir = os.path.dirname(os.path.dirname(os.getcwd()))

sys.path.insert(0, root_dir)

# Import các module từ dự án
from scr.game.ChessBoard import ChessBoard
from scr.game.ChessEngine import ChessEngine
from scr.agents.q_learning_agent import QLearningAgent
from scr.models.advanced_q_network import AdvancedQNetwork
from scr.training.trainer import Trainer
from scr.training.model_evaluator import ModelEvaluator
from scr.data.DataCollector import DataCollector

def get_log_filename():
    """Tạo tên file log theo định dạng yêu cầu"""
    now = datetime.now()
    return f"Train_{now.day}_{now.month}_{now.year}_{now.hour}.log"

def setup_training_environment():
    """Thiết lập môi trường huấn luyện"""
    # Khởi tạo engine (engine sẽ tự tạo bàn cờ)
    engine = ChessEngine()
    
    # Khởi tạo kích thước input/output
    input_size = 64 * 12  # 64 ô, mỗi ô có 12 trạng thái (6 loại quân, 2 màu)
    output_size = 64 * 64  # Số lượng nước đi có thể (từ ô -> đến ô)
    
    # Khởi tạo agent với các tham số
    agent = QLearningAgent(
        state_size=input_size,
        action_size=output_size,
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
    
    # Khởi tạo data collector
    data_collector = DataCollector()
    
    return engine, agent, trainer, evaluator, data_collector

def train_ai(max_moves=100, max_time=300):
    """Huấn luyện AI một ván"""
    # Thiết lập môi trường
    engine, agent, trainer, evaluator, data_collector = setup_training_environment()
    
    # Lưu trữ kết quả huấn luyện
    training_stats = {
        'episode_rewards': [],
        'win_rates': [],
        'epsilon_values': [],
        'losses': [],
        'moves_per_episode': [],
        'time_per_episode': [],
        'agent': agent
    }
    
    print("Bắt đầu huấn luyện AI...")
    
    # Đo thời gian bắt đầu episode
    episode_start = time.time()
    
    # Huấn luyện một episode
    episode_reward, episode_loss, episode_moves = trainer.train_episode(max_moves=max_moves, max_time=max_time)
    
    # Tính thời gian episode
    episode_time = time.time() - episode_start
    
    # Lưu thống kê
    training_stats['episode_rewards'].append(episode_reward)
    training_stats['losses'].append(episode_loss)
    training_stats['epsilon_values'].append(agent.epsilon)
    training_stats['moves_per_episode'].append(episode_moves)
    training_stats['time_per_episode'].append(episode_time)
    
    # Đánh giá kết quả
    win_rate = evaluator.evaluate(num_games=1)
    training_stats['win_rates'].append(win_rate)
    
    # Tạo log chi tiết
    log_data = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'episode_stats': {
            'total_reward': episode_reward,
            'win_rate': win_rate,
            'epsilon': agent.epsilon,
            'avg_loss': episode_loss,
            'moves_count': episode_moves,
            'time_taken': episode_time,
            'final_position': str(engine.board)
        }
    }
    
    # Lưu log
    log_filename = get_log_filename()
    log_path = os.path.join(root_dir, 'logs', log_filename)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    with open(log_path, 'w') as f:
        json.dump(log_data, f, indent=4)
    
    print(f"\nLog đã được lưu tại: {log_path}")
    print("\nKết quả huấn luyện:")
    print(f"Tổng phần thưởng: {episode_reward:.2f}")
    print(f"Tỷ lệ thắng: {win_rate:.2%}")
    print(f"Epsilon: {agent.epsilon:.4f}")
    print(f"Loss trung bình: {episode_loss:.4f}")
    print(f"Số nước đi: {episode_moves}")
    print(f"Thời gian: {episode_time:.1f}s")
    
    return training_stats

def main():
    """Hàm chính"""
    # Tạo thư mục lưu model và logs nếu chưa tồn tại
    model_dir = os.path.join(root_dir, 'models')
    logs_dir = os.path.join(root_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Huấn luyện AI một ván
    training_stats = train_ai(
        max_moves=100,        # Số nước đi tối đa mỗi episode
        max_time=300          # Thời gian tối đa mỗi episode (giây)
    )
    
    # Lưu model
    final_model_path = os.path.join(model_dir, 'chess_ai_final.pth')
    training_stats['agent'].save_model(final_model_path)
    
    print(f"\nModel đã được lưu tại: {final_model_path}")

if __name__ == "__main__":
    main() 