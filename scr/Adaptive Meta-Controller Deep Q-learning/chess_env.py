import numpy as np
import torch
from typing import Tuple, List
from scr.game.ChessEngine import ChessEngine

class ChessEnv:
    def __init__(self):
        """Khởi tạo môi trường cờ vua"""
        self.engine = ChessEngine()
        self.board_size = 8
        self.num_channels = 12  # 6 loại quân * 2 màu
        self.action_size = 64 * 64  # 64 ô nguồn * 64 ô đích
        
    def reset(self) -> torch.Tensor:
        """Reset bàn cờ về trạng thái ban đầu"""
        self.engine = ChessEngine()
        return self._encode_state()
    
    def step(self, action: int) -> Tuple[torch.Tensor, float, bool]:
        """
        Thực hiện một nước đi
        
        Args:
            action (int): Chỉ số hành động (0-4095)
            
        Returns:
            Tuple[torch.Tensor, float, bool]: (trạng thái mới, phần thưởng, done)
        """
        # Chuyển đổi action thành nước đi
        from_square, to_square = self._decode_action(action)
        
        # Thực hiện nước đi
        success, message = self.engine.make_move(from_square, to_square)
        
        # Tính phần thưởng
        reward = self._calculate_reward(success, message)
        
        # Kiểm tra kết thúc
        done = self.engine.is_checkmate() or self.engine.is_stalemate()
        
        return self._encode_state(), reward, done
    
    def _encode_state(self) -> torch.Tensor:
        """
        Chuyển trạng thái bàn cờ thành tensor one-hot encoding
        
        Returns:
            torch.Tensor: Tensor shape (12, 8, 8)
        """
        state = torch.zeros(self.num_channels, self.board_size, self.board_size)
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                piece = self.engine.board.get_piece(row, col)
                if piece:
                    # Tính chỉ số kênh cho quân cờ
                    piece_type = piece.type
                    piece_color = piece.color
                    channel_idx = piece_type + (6 if piece_color == 'white' else 0)
                    
                    # Đặt 1 vào vị trí tương ứng
                    state[channel_idx, row, col] = 1
        
        return state
    
    def _decode_action(self, action: int) -> Tuple[int, int]:
        """
        Chuyển đổi chỉ số hành động thành nước đi
        
        Args:
            action (int): Chỉ số hành động (0-4095)
            
        Returns:
            Tuple[int, int]: (ô nguồn, ô đích)
        """
        from_square = action // 64
        to_square = action % 64
        return from_square, to_square
    
    def _calculate_reward(self, success: bool, message: str) -> float:
        """
        Tính phần thưởng cho nước đi
        
        Args:
            success (bool): Nước đi có thành công không
            message (str): Thông báo từ engine
            
        Returns:
            float: Phần thưởng
        """
        if not success:
            return -10.0  # Phạt nước đi không hợp lệ
            
        if "checkmate" in message:
            return 100.0  # Thưởng chiếu hết
            
        if "check" in message:
            return 5.0  # Thưởng chiếu
            
        if "capture" in message:
            return 2.0  # Thưởng ăn quân
            
        return 0.1  # Thưởng nhỏ cho nước đi hợp lệ 