import torch
import random
import numpy as np
import time
import torch.optim as optim

class Trainer:
    def __init__(self, agent, engine):
        """
        Huấn luyện tác nhân Q-learning.

        Args:
            agent: Đối tượng QLearningAgent cần được huấn luyện
            engine: Đối tượng ChessEngine để tương tác với môi trường cờ vua
        """
        self.agent = agent
        self.engine = engine
        
    def train_episode(self, max_moves=100, max_time=300):
        """
        Huấn luyện một episode.
        
        Args:
            max_moves (int): Số nước đi tối đa cho mỗi episode
            max_time (int): Thời gian tối đa cho mỗi episode (giây)
            
        Returns:
            tuple: (tổng phần thưởng, loss trung bình, số nước đi)
        """
        self.engine.reset()
        state = self.engine.get_board_state_tensor()
        total_reward = 0
        losses = []
        moves_count = 0
        start_time = time.time()
        
        while not self.engine.is_game_over() and moves_count < max_moves:
            # Kiểm tra thời gian
            if time.time() - start_time > max_time:
                print(f"Episode kết thúc do hết thời gian ({max_time}s)")
                break
                
            # Lấy danh sách các nước đi hợp lệ dưới dạng indices
            legal_moves = self.engine.get_legal_moves_as_indices()
            if not legal_moves:
                print("Không còn nước đi hợp lệ")
                break
            
            # Chọn hành động
            action = self.agent.choose_action(state, legal_moves)
            
            # Thực hiện nước đi
            success, message = self.engine.make_move_from_index(action)
            if not success:
                print(f"Lỗi khi thực hiện nước đi: {message}")
                break
                
            # Lấy trạng thái mới và phần thưởng
            next_state = self.engine.get_board_state_tensor()
            reward = self._calculate_reward()
            done = self.engine.is_game_over()
            
            # Lưu transition vào bộ nhớ
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Cập nhật mạng
            loss = self.agent.experience_replay()
            if loss is not None:
                losses.append(loss)
            
            # Cập nhật trạng thái và phần thưởng
            state = next_state
            total_reward += reward
            moves_count += 1
            
            # In tiến trình
            if moves_count % 10 == 0:
                print(f"Episode: {moves_count} nước đi, Reward: {total_reward:.2f}")
        
        # Tính loss trung bình
        avg_loss = np.mean(losses) if losses else 0
        
        # In kết quả episode
        print(f"Episode kết thúc sau {moves_count} nước đi")
        print(f"Tổng phần thưởng: {total_reward:.2f}")
        print(f"Loss trung bình: {avg_loss:.4f}")
        
        return total_reward, avg_loss, moves_count
        
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

    def save_progress(self, filename: str):
        """
        Lưu trạng thái mô hình hiện tại.

        Args:
            filename (str): Đường dẫn file để lưu mô hình.
        """
        torch.save(self.agent.q_network.state_dict(), filename)
        print(f"Model saved to {filename}")
