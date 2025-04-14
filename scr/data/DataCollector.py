import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

class DataCollector:
    """
    Lớp DataCollector để thu thập và lưu trữ dữ liệu huấn luyện AI cờ vua.
    """
    
    def __init__(self):
        """Khởi tạo DataCollector với thư mục lưu trữ mặc định."""
        # Tạo thư mục data/logs nếu chưa tồn tại
        self.base_dir = Path('data/logs')
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
    def save_training_data(self, training_stats: dict):
        """
        Lưu dữ liệu huấn luyện vào file.
        
        Args:
            training_stats (dict): Dictionary chứa các thống kê huấn luyện
                - episode_rewards: List các phần thưởng của mỗi episode
                - win_rates: List tỷ lệ thắng qua các lần đánh giá
                - epsilon_values: List giá trị epsilon
                - losses: List giá trị loss
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Chuyển đổi dữ liệu thành DataFrame
        df = pd.DataFrame({
            'episode_rewards': training_stats['episode_rewards'],
            'losses': training_stats['losses'],
            'epsilon_values': training_stats['epsilon_values']
        })
        
        # Thêm win_rates vào DataFrame (có thể ít hơn số episode)
        if training_stats['win_rates']:
            eval_interval = len(df) // len(training_stats['win_rates'])
            df['win_rates'] = None
            for i, rate in enumerate(training_stats['win_rates']):
                df.loc[i * eval_interval, 'win_rates'] = rate
        
        # Lưu DataFrame vào file CSV
        csv_path = self.base_dir / f'training_stats_{timestamp}.csv'
        df.to_csv(csv_path, index=True)
        print(f"Đã lưu dữ liệu huấn luyện vào: {csv_path}")
        
    def save_game_stats(self, game_stats: dict):
        """
        Lưu thống kê trận đấu.
        
        Args:
            game_stats (dict): Dictionary chứa thống kê trận đấu
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.base_dir / f'game_stats_{timestamp}.json'
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(game_stats, f, indent=4)
        print(f"Đã lưu thống kê trận đấu vào: {json_path}")
        
    def save_model_performance(self, performance_data: dict):
        """
        Lưu dữ liệu về hiệu suất của model.
        
        Args:
            performance_data (dict): Dictionary chứa dữ liệu hiệu suất
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = self.base_dir / f'model_performance_{timestamp}.json'
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(performance_data, f, indent=4)
        print(f"Đã lưu dữ liệu hiệu suất vào: {json_path}")

    def save_ai_data(self, ai_data):
        """Lưu dữ liệu về quá trình học của AI."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.base_dir / f'ai_{timestamp}.log'
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Dữ liệu AI ===\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Giá trị epsilon hiện tại: {ai_data['epsilon'][-1]}\n")
            f.write(f"Phần thưởng nhận được: {ai_data['rewards'][-1]}\n")
            f.write(f"Loss: {ai_data['losses'][-1]}\n")
            f.write(f"AI chọn nước đi tốt nhất: {ai_data['moves'][-1]}\n")
    
    def save_game_strategy(self, strategy_data):
        """Lưu dữ liệu về chiến lược trò chơi."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = self.base_dir / f'game_{timestamp}.log'
        
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"=== Chiến lược trò chơi ===\n")
            f.write(f"Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Số lần kiểm soát trung tâm: {strategy_data['center_control']}\n")
            f.write(f"Số lần phát triển quân: {strategy_data['piece_development']}\n")
            f.write(f"Số lần chiếu: {strategy_data['check_count']}\n")
            f.write(f"Tổng giá trị quân cờ: {strategy_data['captures']}\n")

    def _ensure_data_dir(self):
        """Tạo thư mục lưu trữ dữ liệu nếu chưa tồn tại."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def add_move(self, state_vector, move, reward):
        """
        Thêm một nước đi vào dữ liệu game hiện tại.
        
        Args:
            state_vector (np.ndarray): Vector trạng thái bàn cờ.
            move (str): Nước đi dạng 'e2e4'.
            reward (float): Phần thưởng nhận được.
        """
        self.current_game_data['states'].append(state_vector.tolist())
        self.current_game_data['moves'].append(move)
        self.current_game_data['rewards'].append(float(reward))
        
    def set_game_result(self, winner):
        """
        Cập nhật kết quả ván đấu.
        
        Args:
            winner (str): Người chiến thắng ('white', 'black', hoặc 'draw').
        """
        self.current_game_data['game_result'] = winner
        
    def save_game_data(self):
        """Lưu dữ liệu ván đấu hiện tại vào file."""
        if not self.current_game_data['moves']:
            return  # Không lưu nếu không có nước đi nào
            
        # Thêm timestamp
        self.current_game_data['timestamp'] = datetime.now().isoformat()
        
        # Tạo tên file với timestamp
        filename = f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.data_dir, filename)
        
        # Lưu dữ liệu vào file JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.current_game_data, f, indent=2)
            
        # Reset dữ liệu game
        self.current_game_data = {
            'moves': [],
            'states': [],
            'rewards': [],
            'game_result': None,
            'timestamp': None
        }
        
    @staticmethod
    def load_training_data(data_dir="data/training"):
        """
        Đọc tất cả dữ liệu huấn luyện từ thư mục.
        
        Args:
            data_dir (str): Thư mục chứa dữ liệu huấn luyện.
            
        Returns:
            tuple: (states, moves, rewards, results) - Dữ liệu huấn luyện đã được xử lý.
        """
        all_states = []
        all_moves = []
        all_rewards = []
        all_results = []
        
        if not os.path.exists(data_dir):
            return np.array([]), [], np.array([]), []
            
        for filename in os.listdir(data_dir):
            if not filename.endswith('.json'):
                continue
                
            filepath = os.path.join(data_dir, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                game_data = json.load(f)
                
            all_states.extend(game_data['states'])
            all_moves.extend(game_data['moves'])
            all_rewards.extend(game_data['rewards'])
            all_results.extend([game_data['game_result']] * len(game_data['moves']))
            
        return (np.array(all_states), all_moves, 
                np.array(all_rewards), all_results)
                
    def get_statistics(self):
        """
        Tính toán thống kê từ dữ liệu đã thu thập.
        
        Returns:
            dict: Các thống kê về dữ liệu huấn luyện.
        """
        all_data = self.load_training_data(self.data_dir)
        states, moves, rewards, results = all_data
        
        if len(states) == 0:
            return {
                'total_games': 0,
                'total_moves': 0,
                'avg_moves_per_game': 0,
                'white_wins': 0,
                'black_wins': 0,
                'draws': 0,
                'avg_reward': 0,
                'max_reward': 0,
                'min_reward': 0
            }
            
        # Tính số ván đấu duy nhất
        unique_timestamps = set()
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.json'):
                with open(os.path.join(self.data_dir, filename), 'r') as f:
                    game_data = json.load(f)
                    if game_data['timestamp']:
                        unique_timestamps.add(game_data['timestamp'])
                        
        total_games = len(unique_timestamps)
        
        # Tính thống kê khác
        stats = {
            'total_games': total_games,
            'total_moves': len(moves),
            'avg_moves_per_game': len(moves) / total_games if total_games > 0 else 0,
            'white_wins': results.count('white'),
            'black_wins': results.count('black'),
            'draws': results.count('draw'),
            'avg_reward': float(np.mean(rewards)),
            'max_reward': float(np.max(rewards)),
            'min_reward': float(np.min(rewards))
        }
        
        return stats 