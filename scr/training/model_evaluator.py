import torch
import numpy as np
from scr.game.ChessEngine import ChessEngine
import logging

class ModelEvaluator:
    def __init__(self, agent, engine):
        """
        Đánh giá hiệu suất của tác nhân sau khi huấn luyện.

        Args:
            agent: Đối tượng QLearningAgent cần đánh giá
            engine: Đối tượng ChessEngine để tương tác với môi trường cờ vua
        """
        self.agent = agent
        self.engine = engine
        self.total_rewards = []
        self.logger = logging.getLogger(__name__)

    def evaluate(self, num_games=10):
        """Đánh giá mô hình bằng cách chơi một số ván"""
        wins = 0
        total_games = num_games
        
        for game in range(num_games):
            self.engine.reset()
            done = False
            moves = 0
            max_moves = 100  # Giới hạn số nước đi tối đa
            
            while not done and moves < max_moves:
                # Lấy trạng thái hiện tại
                state = self.engine.get_board_state_tensor()
                
                # Lấy danh sách nước đi hợp lệ
                legal_moves = self.engine.get_legal_moves_as_indices()
                
                if not legal_moves:
                    self.logger.warning("Không còn nước đi hợp lệ")
                    break
                
                # Chọn nước đi (không khám phá ngẫu nhiên)
                action = self.agent.choose_action(state, legal_moves)
                
                # Thực hiện nước đi
                try:
                    self.engine.make_move_from_index(action)
                    moves += 1
                except Exception as e:
                    self.logger.error(f"Lỗi khi thực hiện nước đi: {str(e)}")
                    break
                
                # Kiểm tra kết thúc game
                if self.engine.is_game_over():
                    done = True
                    if self.engine.is_checkmate():
                        wins += 1
                    break
            
            self.logger.info(f"Game {game + 1}/{num_games} kết thúc sau {moves} nước đi")
        
        win_rate = wins / total_games
        self.logger.info(f"Tỷ lệ thắng: {win_rate:.2%}")
        return win_rate
        
    def _calculate_reward(self):
        """
        Tính toán phần thưởng dựa trên trạng thái hiện tại của bàn cờ.
        
        Returns:
            float: Giá trị phần thưởng
        """
        if self.engine.is_game_over():
            if self.engine.is_checkmate():
                return 100 if not self.engine.board.turn else -100
            else:
                return 0  # Hòa
                
        # Phần thưởng cho việc kiểm soát trung tâm
        reward = 0
        center_squares = ['d4', 'd5', 'e4', 'e5']
        for square in center_squares:
            piece = self.engine.get_piece_at(square)
            if piece:
                reward += 1 if piece.isupper() else -1
                
        # Phần thưởng cho việc chiếu
        if self.engine.is_in_check():
            reward += 5 if not self.engine.board.turn else -5
            
        return reward

    def generate_report(self, filename: str):
        """
        Xuất kết quả đánh giá vào file.

        Args:
            filename (str): Tên file để lưu báo cáo.
        """
        with open(filename, 'w') as f:
            f.write(f"Evaluation Results:\n")
            f.write(f"Average Reward: {np.mean(self.total_rewards)}\n")
        print(f"Report saved to {filename}")
