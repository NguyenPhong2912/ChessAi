import pygame
import sys
import torch
import numpy as np
import os
import logging
from datetime import datetime
from scr.game.ChessBoard import ChessBoard
from scr.agents.q_learning_agent import QLearningAgent
from scr.models.q_network import QNetwork
from scr.game.ChessRules import ChessRules
from scr.utils.callback_logger import CallbackLogger
from scr.utils.move_encoder import MoveEncoder
import random
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class EloRating:
    def __init__(self, k_factor=32):
        """
        Khởi tạo hệ thống Elo rating.
        
        Args:
            k_factor (float): Hệ số K cho tính toán Elo (mặc định: 32)
        """
        self.k_factor = k_factor
        self.white_rating = 1200  # Rating ban đầu cho người chơi trắng
        self.black_rating = 1200  # Rating ban đầu cho người chơi đen
        self.rating_history = []

    def update_ratings(self, winner):
        """
        Cập nhật Elo rating sau mỗi ván đấu.
        
        Args:
            winner (str): 'white', 'black', hoặc 'draw'
        """
        # Tính expected score
        expected_white = 1 / (1 + 10 ** ((self.black_rating - self.white_rating) / 400))
        expected_black = 1 - expected_white

        # Tính actual score
        if winner == 'white':
            actual_white = 1
            actual_black = 0
        elif winner == 'black':
            actual_white = 0
            actual_black = 1
        else:  # draw
            actual_white = 0.5
            actual_black = 0.5

        # Cập nhật rating
        self.white_rating += self.k_factor * (actual_white - expected_white)
        self.black_rating += self.k_factor * (actual_black - expected_black)

        # Lưu lịch sử rating
        self.rating_history.append({
            'white': round(self.white_rating),
            'black': round(self.black_rating),
            'winner': winner
        })

    def get_ratings(self):
        """
        Lấy rating hiện tại của cả hai người chơi.
        
        Returns:
            tuple: (white_rating, black_rating)
        """
        return round(self.white_rating), round(self.black_rating)

class ChessVisualizer:
    # Mapping từ ký tự quân cờ sang tên file hình ảnh
    PIECE_IMAGES = {
        'K': 'white_king.png',
        'Q': 'white_queen.png',
        'R': 'white_rook.png',
        'B': 'white_bishop.png',
        'N': 'white_knight.png',
        'P': 'white_pawn.png',
        'k': 'black_king.png',
        'q': 'black_queen.png',
        'r': 'black_rook.png',
        'b': 'black_bishop.png',
        'n': 'black_knight.png',
        'p': 'black_pawn.png'
    }

    def __init__(self, chess_board, agent, q_network, output_dir):
        """
        Khởi tạo bộ trực quan hóa bàn cờ.
        
        Args:
            chess_board (ChessBoard): Đối tượng bàn cờ để hiển thị.
            agent (QLearningAgent): Đối tượng AI agent.
            q_network (QNetwork): Mạng nơ-ron Q-learning.
            output_dir (str): Đường dẫn đến thư mục đầu ra để lưu trữ các biểu đồ.
        """
        self.chess_board = chess_board
        self.agent = agent
        self.q_network = q_network
        
        # Khởi tạo các module utils
        self.callback_logger = CallbackLogger('chess_training.log')
        self.move_encoder = MoveEncoder()
        
        # Khởi tạo data collector
        from scr.data.DataCollector import DataCollector
        self.data_collector = DataCollector()
        
        pygame.init()
        self.square_size = 80  # Tăng kích thước ô để phù hợp với hình ảnh 80px
        self.board_size = self.square_size * 8
        self.screen = pygame.display.set_mode((self.board_size, self.board_size))
        pygame.display.set_caption('Cờ Vua - AI')
        
        # Load hình ảnh quân cờ
        self.piece_images = {}
        self.load_piece_images()
        
        self.selected_square = None
        self.dragging = False
        self.drag_piece = None
        self.drag_pos = None
        self.drag_from_pos = None
        self.current_turn = 'white'
        self.game_mode = 'human_vs_ai'
        
        # Khởi tạo các biến theo dõi trò chơi
        self.epoch = 1
        self.total_games = 0
        self.wins = {'white': 0, 'black': 0, 'draws': 0}
        self.elo = EloRating()
        self.current_loss = 0.0
        self.loss_history = []
        
        # Khởi tạo logging trước
        self.setup_logging()
        
        # Khởi tạo database PGN
        self.setup_pgn_database()
        
        # Log thông tin khởi động
        self.game_logger.info("Khởi động trò chơi mới")
        self.game_logger.info(f"Chế độ chơi: {self.game_mode}")
        self.game_logger.info(f"Epoch: {self.epoch}")
        
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def setup_logging(self):
        """
        Thiết lập logging cho trò chơi.
        """
        # Tạo thư mục logs nếu chưa tồn tại
        logs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'data', 'logs')
        os.makedirs(logs_dir, exist_ok=True)
        
        # Tạo timestamp cho tất cả các file log
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Tạo các logger riêng biệt
        self.game_logger = self._create_logger('game', os.path.join(logs_dir, f'game_{timestamp}.log'))
        self.ai_logger = self._create_logger('ai', os.path.join(logs_dir, f'ai_{timestamp}.log'))
        self.stats_logger = self._create_logger('stats', os.path.join(logs_dir, f'stats_{timestamp}.log'))
        self.error_logger = self._create_logger('error', os.path.join(logs_dir, f'error_{timestamp}.log'))
        
        # Log thông tin khởi động
        self.game_logger.info("="*50)
        self.game_logger.info("Khởi động trò chơi mới")
        self.game_logger.info(f"Chế độ chơi: {self.game_mode}")
        self.game_logger.info(f"Epoch: {self.epoch}")
        self.game_logger.info("="*50)

    def _create_logger(self, name, log_file):
        """
        Tạo logger với tên và file log cụ thể.
        
        Args:
            name (str): Tên của logger
            log_file (str): Đường dẫn đến file log
            
        Returns:
            logging.Logger: Logger đã được cấu hình
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        
        # Xóa các handler cũ nếu có
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Handler cho file
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Handler cho console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Format cho log
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Thêm handlers
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger

    def setup_pgn_database(self):
        """
        Thiết lập database PGN cho việc lưu trữ các ván cờ.
        """
        # Tạo thư mục database trong thư mục gốc của dự án
        self.pgn_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 'database')
        os.makedirs(self.pgn_dir, exist_ok=True)
        
        # Khởi tạo danh sách nước đi cho ván cờ hiện tại
        self.current_game_moves = []
        self.current_game_start_time = datetime.now()
        
        self.game_logger.info(f"Đã khởi tạo database tại: {self.pgn_dir}")

    def save_game_to_pgn(self, winner):
        """
        Lưu ván cờ hiện tại vào file PGN.
        
        Args:
            winner (str): 'white', 'black', hoặc 'draw'
        """
        try:
            # Tạo tên file PGN với timestamp
            timestamp = self.current_game_start_time.strftime('%Y%m%d_%H%M%S')
            pgn_file = os.path.join(self.pgn_dir, f'game_{timestamp}.pgn')
            
            # Tạo header PGN
            pgn_content = [
                '[Event "Chess Game"]',
                f'[Date "{self.current_game_start_time.strftime("%Y.%m.%d")}"]',
                f'[Time "{self.current_game_start_time.strftime("%H:%M:%S")}"]',
                '[White "Human"]',
                '[Black "AI"]',
                f'[Result "{self._get_pgn_result(winner)}"]',
                '',
            ]
            
            # Thêm các nước đi
            moves = []
            for i, move in enumerate(self.current_game_moves, 1):
                if i % 2 == 1:  # Nước đi của trắng
                    moves.append(f"{i}. {move}")
                else:  # Nước đi của đen
                    moves.append(move)
            
            # Thêm kết quả cuối cùng
            moves.append(self._get_pgn_result(winner))
            
            # Gộp tất cả nước đi thành một chuỗi
            pgn_content.append(' '.join(moves))
            
            # Lưu vào file
            with open(pgn_file, 'w', encoding='utf-8') as f:
                f.write('\n'.join(pgn_content))
            
            self.game_logger.info(f"Đã lưu ván cờ vào file: {pgn_file}")
            
        except Exception as e:
            self.game_logger.error(f"Lỗi khi lưu file PGN: {str(e)}")

    def _get_pgn_result(self, winner):
        """
        Chuyển đổi kết quả sang định dạng PGN.
        
        Args:
            winner (str): 'white', 'black', hoặc 'draw'
            
        Returns:
            str: Kết quả theo định dạng PGN
        """
        if winner == 'white':
            return '1-0'
        elif winner == 'black':
            return '0-1'
        else:
            return '1/2-1/2'

    def load_game_from_pgn(self, pgn_file):
        """
        Load một ván cờ từ file PGN.
        
        Args:
            pgn_file (str): Đường dẫn đến file PGN
        """
        try:
            with open(pgn_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Tách header và moves
            header_end = content.find('\n\n')
            if header_end == -1:
                raise ValueError("File PGN không hợp lệ")
            
            # Lấy phần moves
            moves = content[header_end + 2:].strip()
            
            # Tách các nước đi
            move_list = moves.split()
            
            # Reset bàn cờ
            self.chess_board.reset()
            
            # Thực hiện lại các nước đi
            for move in move_list:
                if move not in ['1-0', '0-1', '1/2-1/2']:  # Bỏ qua kết quả
                    self.chess_board.apply_move(move)
            
            self.game_logger.info(f"Đã load ván cờ từ file: {pgn_file}")
            
        except Exception as e:
            self.game_logger.error(f"Lỗi khi load file PGN: {str(e)}")

    def load_piece_images(self):
        """
        Load hình ảnh quân cờ từ thư mục assets.
        """
        assets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets')
        images_dir = os.path.join(assets_dir, 'images', 'imgs-80px')
        
        for piece, filename in self.PIECE_IMAGES.items():
            image_path = os.path.join(images_dir, filename)
            try:
                image = pygame.image.load(image_path)
                # Scale hình ảnh để vừa với ô bàn cờ
                image = pygame.transform.scale(image, (self.square_size, self.square_size))
                self.piece_images[piece] = image
            except pygame.error as e:
                print(f"Không thể load hình ảnh {filename}: {e}")

    def draw_board(self):
        """
        Vẽ bàn cờ vua bằng Pygame.
        """
        self.screen.fill((255, 255, 255))  # Nền trắng

        # Vẽ ô bàn cờ
        for row in range(8):
            for col in range(8):
                color = (240, 240, 240) if (row + col) % 2 == 0 else (120, 120, 120)
                pygame.draw.rect(self.screen, color, 
                               (col * self.square_size, row * self.square_size, 
                                self.square_size, self.square_size))

        # Vẽ gợi ý nước đi hợp lệ
        if self.selected_square:
            from_pos = self.get_chess_pos(self.selected_square)
            board_state = self.chess_board.get_state()
            
            # Duyệt qua tất cả các ô trên bàn cờ
            for to_row in range(8):
                for to_col in range(8):
                    to_square = (to_col, to_row)
                    to_pos = self.get_chess_pos(to_square)
                    move = f"{from_pos}{to_pos}"
                    
                    # Kiểm tra nước đi hợp lệ
                    if ChessRules.is_valid_move(board_state, move, self.current_turn):
                        # Vẽ gợi ý nước đi (hình tròn màu xanh nhạt)
                        center = (to_col * self.square_size + self.square_size/2,
                                to_row * self.square_size + self.square_size/2)
                        # Nếu có quân đối phương ở ô đích, vẽ viền đỏ
                        if to_pos in board_state:
                            pygame.draw.circle(self.screen, (255, 0, 0), center, self.square_size/4, 3)
                        else:
                            pygame.draw.circle(self.screen, (0, 255, 0, 128), center, self.square_size/6)
                
        # Vẽ viền cho ô được chọn
        if self.selected_square:
            col, row = self.selected_square
            pygame.draw.rect(self.screen, (255, 255, 0),
                           (col * self.square_size, row * self.square_size,
                            self.square_size, self.square_size), 3)

        # Vẽ quân cờ
        board_state = self.chess_board.get_state()
        if isinstance(board_state, dict):
            for pos, piece in board_state.items():
                if piece in self.piece_images:
                    col = ord(pos[0]) - ord('a')
                    row = 7 - (int(pos[1]) - 1)
                    # Không vẽ quân cờ đang được kéo
                    if not (self.dragging and self.drag_from_pos == pos):
                        # Vẽ hình ảnh quân cờ
                        piece_image = self.piece_images[piece]
                        piece_rect = piece_image.get_rect(center=(col * self.square_size + self.square_size/2,
                                                                row * self.square_size + self.square_size/2))
                        self.screen.blit(piece_image, piece_rect)
                    
                    # Debug: Vẽ tọa độ của mỗi quân cờ
                    font = pygame.font.SysFont('Arial', 12)
                    text = font.render(pos, True, (255, 0, 0))
                    self.screen.blit(text, (col * self.square_size + 5, row * self.square_size + 5))

        # Vẽ quân cờ đang được kéo
        if self.dragging and self.drag_piece:
            piece_image = self.piece_images[self.drag_piece]
            # Lấy vị trí chuột làm tâm của quân cờ
            piece_rect = piece_image.get_rect(center=self.drag_pos)
            self.screen.blit(piece_image, piece_rect)

    def get_square_from_pos(self, pos):
        """
        Chuyển đổi vị trí chuột thành tọa độ ô trên bàn cờ.
        """
        x, y = pos
        # Đảm bảo tọa độ nằm trong bàn cờ
        x = max(0, min(x, self.board_size - 1))
        y = max(0, min(y, self.board_size - 1))
        col = x // self.square_size
        row = y // self.square_size
        print(f"Click position: {pos}, Square: ({col}, {row})")  # Debug log
        return (col, row)

    def get_pos_from_square(self, square):
        """
        Chuyển đổi tọa độ ô trên bàn cờ thành vị trí chuột.
        """
        col, row = square
        return (col * self.square_size + self.square_size/2,
                row * self.square_size + self.square_size/2)

    def get_chess_pos(self, square):
        """
        Chuyển đổi tọa độ ô thành ký hiệu vị trí cờ vua (ví dụ: 'e2').
        """
        col, row = square
        # Chuyển đổi tọa độ Pygame sang tọa độ cờ vua
        file = chr(ord('a') + col)  # Cột (a-h)
        rank = str(8 - row)  # Hàng (1-8)
        pos = f"{file}{rank}"
        print(f"Square to chess pos: {square} -> {pos}")  # Debug log
        return pos

    def get_state_vector(self):
        """
        Chuyển đổi trạng thái bàn cờ thành vector đầu vào cho mạng nơ-ron.
        """
        board_state = self.chess_board.get_state()
        state_vector = np.zeros(64 * 12)  # 64 ô, mỗi ô có 12 trạng thái (6 loại quân * 2 màu)
        
        # Mã hóa vị trí và loại quân cờ
        for pos, piece in board_state.items():
            col = ord(pos[0]) - ord('a')  # 0-7 cho a-h
            row = int(pos[1]) - 1  # 0-7 cho 1-8
            square_idx = row * 8 + col
            
            # Mã hóa loại quân cờ
            piece_type = piece.upper()
            color_idx = 0 if piece.isupper() else 6  # 0-5 cho trắng, 6-11 cho đen
            piece_idx = "KQRBNP".index(piece_type)
            
            # Đặt giá trị 1 cho vị trí tương ứng
            state_vector[square_idx * 12 + color_idx + piece_idx] = 1
            
        return torch.FloatTensor(state_vector).unsqueeze(0)

    def get_legal_moves(self):
        """
        Lấy danh sách các nước đi hợp lệ cho lượt hiện tại.
        """
        moves = []
        board_state = self.chess_board.get_state()
        
        # Duyệt qua tất cả các ô trên bàn cờ
        for from_file in range(8):  # a-h
            for from_rank in range(1, 9):  # 1-8
                from_pos = f"{chr(ord('a') + from_file)}{from_rank}"
                
                # Kiểm tra có quân cờ ở ô nguồn không
                if from_pos not in board_state:
                    continue
                    
                piece = board_state[from_pos]
                is_white_piece = piece.isupper()
                
                # Chỉ xét các quân cờ của lượt hiện tại
                if (self.current_turn == 'white' and not is_white_piece) or \
                   (self.current_turn == 'black' and is_white_piece):
                    continue
                
                # Duyệt qua tất cả các ô đích có thể
                for to_file in range(8):  # a-h
                    for to_rank in range(1, 9):  # 1-8
                        to_pos = f"{chr(ord('a') + to_file)}{to_rank}"
                        move = f"{from_pos}{to_pos}"
                        
                        # Kiểm tra nước đi có hợp lệ không
                        if ChessRules.is_valid_move(board_state, move, self.current_turn):
                            # Chuyển đổi nước đi thành chỉ số hành động
                            from_idx = from_file + (from_rank - 1) * 8
                            to_idx = to_file + (to_rank - 1) * 8
                            action_idx = from_idx * 64 + to_idx
                            moves.append((move, action_idx))
        
        return moves

    def calculate_loss(self, state_tensor, action_idx):
        """
        Tính toán loss cho mạng nơ-ron.
        """
        with torch.no_grad():
            target_q = self.q_network(state_tensor)[0, action_idx]
            current_q = self.q_network(state_tensor)[0, action_idx]
            loss = torch.nn.functional.mse_loss(current_q, target_q)
            self.current_loss = loss.item()
            self.loss_history.append(self.current_loss)
            return loss

    def calculate_reward(self, move):
        """
        Tính toán phần thưởng cho nước đi.
        
        Args:
            move (str): Nước đi cần tính phần thưởng
            
        Returns:
            float: Phần thưởng cho nước đi
        """
        return self.get_reward()

    def update_game_stats(self, winner):
        """
        Cập nhật thống kê trò chơi.
        
        Args:
            winner (str): 'white', 'black', hoặc 'draw'
        """
        self.total_games += 1
        if winner == 'draw':
            self.wins['draws'] += 1
        else:
            self.wins[winner] += 1
        
        # Cập nhật Elo rating
        self.elo.update_ratings(winner)
        
        # Log thống kê
        white_rating, black_rating = self.elo.get_ratings()
        
        # Tính loss trung bình an toàn
        avg_loss = np.mean(self.loss_history) if self.loss_history else 0.0
        
        stats_message = f"""
Kết thúc ván {self.total_games}:
Người thắng: {winner}
Elo Rating - Trắng: {white_rating}, Đen: {black_rating}
Thống kê: Trắng {self.wins['white']} - Đen {self.wins['black']} - Hòa {self.wins['draws']}
Loss trung bình: {avg_loss:.4f}
"""
        self.stats_logger.info(stats_message)
        
        # Log thông tin ván đấu
        self.callback_logger.log_epoch(
            epoch=self.epoch,
            total_reward=sum(self.loss_history) if self.loss_history else 0.0,
            epsilon=self.agent.epsilon
        )
        
        # Reset loss history cho ván mới
        self.loss_history = []
        self.current_loss = 0.0

    def make_ai_move(self):
        """
        Thực hiện nước đi của AI sử dụng Q-learning và DNN.
        """
        self.ai_logger.info("=== AI's Turn ===")
        self.ai_logger.info("Bắt đầu lượt đi của AI")
        
        # Lấy trạng thái hiện tại của bàn cờ
        state_vector = self.get_state_vector()
        self.ai_logger.info(f"Shape của state_vector: {state_vector.shape}")
        
        legal_moves = self.get_legal_moves()
        self.ai_logger.info(f"Số nước đi hợp lệ: {len(legal_moves)}")
        
        # Log trạng thái bàn cờ chi tiết
        board_state = self.chess_board.get_state()
        self.ai_logger.info("Trạng thái bàn cờ:")
        for pos, piece in board_state.items():
            self.ai_logger.info(f"{piece} tại {pos}")
        
        # Phân tích giá trị quân cờ
        piece_values = {
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100,
            'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -100
        }
        total_value = 0
        for piece in board_state.values():
            total_value += piece_values.get(piece, 0)
        self.ai_logger.info(f"Tổng giá trị quân cờ: {total_value}")
        
        if not legal_moves:
            self.ai_logger.warning("AI không có nước đi hợp lệ!")
            return
        
        # Epsilon-greedy strategy
        epsilon = self.agent.epsilon
        self.ai_logger.info(f"Giá trị epsilon hiện tại: {epsilon}")
        
        if np.random.random() < epsilon:
            # Thăm dò: chọn ngẫu nhiên một nước đi hợp lệ
            move, action_idx = random.choice(legal_moves)
            self.ai_logger.info(f"AI thăm dò - Chọn nước đi ngẫu nhiên: {move}")
        else:
            # Tận dụng: chọn nước đi có Q-value cao nhất
            self.ai_logger.info("AI tận dụng - Tính toán Q-values")
            with torch.no_grad():
                q_values = self.q_network(state_vector)
                self.ai_logger.info(f"Shape của q_values: {q_values.shape}")
                
                # Lọc ra các Q-value của các nước đi hợp lệ
                move_q_values = []
                for move, action_idx in legal_moves:
                    q_value = q_values[0, action_idx].item()
                    move_q_values.append((move, q_value))
                    self.ai_logger.info(f"Nước đi {move} có Q-value: {q_value:.4f}")
                
                # Chọn nước đi có Q-value cao nhất
                best_move, best_q_value = max(move_q_values, key=lambda x: x[1])
                move = best_move
                action_idx = next(idx for m, idx in legal_moves if m == move)
                self.ai_logger.info(f"AI chọn nước đi tốt nhất: {move} (Q-value: {best_q_value:.4f})")
        
        try:
            self.ai_logger.info(f"Thực hiện nước đi: {move}")
            
            # Lưu trạng thái cũ
            old_state_vector = state_vector.clone()
            self.ai_logger.info("Đã lưu trạng thái cũ")
            
            # Thực hiện nước đi
            self.chess_board.apply_move(move)
            self.ai_logger.info("Nước đi thành công")
            
            # Lấy trạng thái mới và phần thưởng
            new_state_vector = self.get_state_vector()
            reward = self.get_reward()
            self.ai_logger.info(f"Phần thưởng nhận được: {reward}")
            
            # Cập nhật Q-value
            self.ai_logger.info("Cập nhật Q-learning")
            self.ai_logger.info("Lưu transition vào memory")
            
            # Mã hóa nước đi
            encoded_move = self.move_encoder.encode(move)
            
            self.agent.store_transition(
                old_state_vector.numpy(),
                encoded_move,  # Sử dụng nước đi đã mã hóa
                reward,
                new_state_vector.numpy(),
                self.chess_board.is_checkmate()
            )
            
            # Học từ experience replay
            self.ai_logger.info("Thực hiện experience replay")
            self.agent.experience_replay()
            
            # Giảm epsilon
            if self.agent.epsilon > self.agent.epsilon_min:
                old_epsilon = self.agent.epsilon
                self.agent.epsilon *= self.agent.epsilon_decay
                self.ai_logger.info(f"Giảm epsilon: {old_epsilon:.4f} -> {self.agent.epsilon:.4f}")
            
            # Kiểm tra chiếu hết
            if self.chess_board.is_checkmate():
                self.ai_logger.info("Phát hiện chiếu hết!")
                winner = 'black'  # AI luôn là quân đen
                self.update_game_stats(winner)
                return
            
            self.current_turn = 'white'
            self.ai_logger.info("Chuyển lượt cho người chơi (trắng)")
            
        except ValueError as e:
            self.error_logger.error(f"Lỗi khi thực hiện nước đi: {str(e)}")
            
        self.ai_logger.info("Kết thúc lượt đi của AI")

    def get_reward(self):
        """
        Tính toán phần thưởng cho AI dựa trên trạng thái bàn cờ.
        """
        board_state = self.chess_board.get_state()
        reward = 0
        
        # Giá trị các quân cờ
        piece_values = {
            'P': 1, 'N': 3, 'B': 3, 'R': 5, 'Q': 9, 'K': 100,
            'p': -1, 'n': -3, 'b': -3, 'r': -5, 'q': -9, 'k': -100
        }
        
        # Tính tổng giá trị quân cờ
        for piece in board_state.values():
            reward += piece_values.get(piece, 0)
        
        # Thưởng thêm cho các tình huống đặc biệt
        if self.chess_board.is_checkmate():
            if self.current_turn == 'black':  # AI thắng
                reward += 1000
                self.game_logger.info("Thưởng +1000 cho chiếu hết")
            else:  # AI thua
                reward -= 1000
                self.game_logger.info("Phạt -1000 cho bị chiếu hết")
        elif self.chess_board.is_check():
            if self.current_turn == 'black':  # AI bị chiếu
                reward -= 10
                self.game_logger.info("Phạt -10 cho bị chiếu")
            else:  # AI chiếu đối thủ
                reward += 10
                self.game_logger.info("Thưởng +10 cho chiếu đối thủ")
        
        # Thưởng cho việc kiểm soát trung tâm
        center_squares = ['d4', 'd5', 'e4', 'e5']
        for square in center_squares:
            if square in board_state:
                piece = board_state[square]
                if piece.islower():  # Quân đen (AI)
                    reward += 2
                    self.game_logger.info(f"Thưởng +2 cho kiểm soát ô {square}")
                else:  # Quân trắng
                    reward -= 2
                    self.game_logger.info(f"Phạt -2 cho mất kiểm soát ô {square}")
        
        # Thưởng cho việc phát triển quân
        development_squares = ['b1', 'g1', 'b8', 'g8']  # Vị trí xuất phát của mã
        for square in development_squares:
            if square in board_state:
                piece = board_state[square]
                if piece.islower() and piece.upper() == 'N':  # Mã đen đã di chuyển
                    reward += 1
                    self.game_logger.info(f"Thưởng +1 cho phát triển mã")
                elif piece.isupper() and piece == 'N':  # Mã trắng chưa di chuyển
                    reward -= 1
                    self.game_logger.info(f"Phạt -1 cho chưa phát triển mã")
                
        self.game_logger.info(f"Tổng phần thưởng: {reward}")
        return reward

    def select_promotion_piece(self) -> str:
        """
        Hiển thị menu và cho phép người chơi chọn quân phong cấp.
        
        Returns:
            str: Ký tự đại diện cho quân được chọn (q, r, b, n)
        """
        promotion_pieces = ['q', 'r', 'b', 'n']
        piece_names = {
            'q': 'Hậu',
            'r': 'Xe',
            'b': 'Tượng',
            'n': 'Mã'
        }
        print("\nChọn quân phong cấp:")
        for piece in promotion_pieces:
            print(f"{piece}: {piece_names[piece]}")
        
        while True:
            choice = input("Nhập lựa chọn (q/r/b/n): ").lower()
            if choice in promotion_pieces:
                return choice
            print("Lựa chọn không hợp lệ. Vui lòng chọn lại.")

    def handle_click(self, pos):
        """
        Xử lý sự kiện click chuột.
        """
        # Nếu đang là lượt của AI và đang chơi với AI, không cho phép người chơi di chuyển
        if self.current_turn == 'black' and self.game_mode == 'human_vs_ai':
            return
            
        square = self.get_square_from_pos(pos)
        chess_pos = self.get_chess_pos(square)
        board_state = self.chess_board.get_state()

        # Log trạng thái cho AI
        if self.current_turn == 'black':
            self.game_logger.info(f"Current turn: {self.current_turn}")
            self.game_logger.info(f"Selected square: {self.selected_square}")
            self.game_logger.info(f"Board state at {chess_pos}: {board_state.get(chess_pos)}")

        # Kiểm tra xem có đang bị chiếu không
        is_in_check = ChessRules.is_check(board_state, self.current_turn)
        if is_in_check and self.current_turn == 'black':
            self.game_logger.info(f"{self.current_turn} đang bị chiếu!")

        if not self.dragging:
            # Chọn quân cờ
            if chess_pos in board_state:
                piece = board_state[chess_pos]
                is_white_piece = piece.isupper()
                is_white_turn = self.current_turn == 'white'
                
                # Log cho AI
                if self.current_turn == 'black':
                    self.game_logger.info(f"Piece: {piece}, Is white piece: {is_white_piece}, Is white turn: {is_white_turn}")
                
                # Kiểm tra lượt đi
                if (is_white_piece and is_white_turn) or (not is_white_piece and not is_white_turn):
                    # Nếu đang bị chiếu, ưu tiên di chuyển vua
                    if is_in_check:
                        king = 'K' if is_white_turn else 'k'
                        if piece.upper() != 'K':
                            # Kiểm tra xem quân này có thể chặn/ăn quân chiếu không
                            can_block_check = False
                            for to_file in range(8):
                                for to_rank in range(1, 9):
                                    to_pos = f"{chr(ord('a') + to_file)}{to_rank}"
                                    move = f"{chess_pos}{to_pos}"
                                    
                                    # Thử nước đi và kiểm tra có thoát chiếu không
                                    if ChessRules.is_valid_move(board_state, move, self.current_turn):
                                        # Thử thực hiện nước đi
                                        new_board = board_state.copy()
                                        new_board[to_pos] = new_board.pop(chess_pos)
                                        
                                        # Kiểm tra sau nước đi này có còn bị chiếu không
                                        if not ChessRules.is_check(new_board, self.current_turn):
                                            can_block_check = True
                                            break
                                if can_block_check:
                                    break
                                    
                            if not can_block_check:
                                if self.current_turn == 'black':
                                    self.game_logger.info("Đang bị chiếu! Phải di chuyển vua hoặc chặn/ăn quân chiếu!")
                                return
                    
                    self.selected_square = square
                    self.dragging = True
                    self.drag_piece = piece
                    self.drag_pos = pos
                    self.drag_from_pos = chess_pos
                    if self.current_turn == 'black':
                        self.game_logger.info(f"Selected piece: {piece} at {chess_pos}")
        else:
            # Thả quân cờ
            if self.drag_from_pos == chess_pos:  # Nếu thả tại vị trí cũ
                self.dragging = False
                self.drag_piece = None
                self.drag_from_pos = None
                self.selected_square = None
                return
                
            move = f"{self.drag_from_pos}{chess_pos}"
            
            # Kiểm tra phong cấp cho quân lính
            if self.drag_piece.upper() == 'P':
                # Kiểm tra quân lính đã đến hàng cuối chưa
                if (self.drag_piece.isupper() and chess_pos[1] == '8') or \
                   (self.drag_piece.islower() and chess_pos[1] == '1'):
                    # Chọn quân phong cấp
                    promotion = self.select_promotion_piece()
                    move += promotion
            
            # Log cho AI
            if self.current_turn == 'black':
                self.game_logger.info(f"Attempting move: {move}")
            
            try:
                # Kiểm tra nước đi hợp lệ
                if ChessRules.is_valid_move(board_state, move, self.current_turn):
                    # Lưu trạng thái trước khi di chuyển
                    state_before = self.get_state_vector()
                    
                    # Thực hiện nước đi
                    self.chess_board.apply_move(move)
                    self.game_logger.info(f"Move successful: {move}")
                    
                    # Lưu nước đi vào danh sách
                    self.current_game_moves.append(move)
                    
                    # Nếu là lượt của AI, tính reward và loss
                    if self.current_turn == 'black':
                        reward = self.calculate_reward(move)
                        state_tensor = self.get_state_vector()
                        action_idx = (ord(self.drag_from_pos[0]) - ord('a') + (int(self.drag_from_pos[1]) - 1) * 8) * 64 + \
                                    (ord(chess_pos[0]) - ord('a') + (int(chess_pos[1]) - 1) * 8)
                        loss = self.calculate_loss(state_tensor, action_idx)
                        self.game_logger.info(f"Reward: {reward:.4f}, Loss: {loss:.4f}")
                        
                        # Lưu dữ liệu huấn luyện
                        self.data_collector.add_move(state_before, move, reward)
                    
                    # Kiểm tra chiếu hết
                    if self.chess_board.is_checkmate():
                        winner = self.current_turn
                        self.game_logger.info(f"Checkmate! {winner} wins!")
                        
                        # Lưu dữ liệu huấn luyện
                        self.data_collector.set_game_result(winner)
                        self.data_collector.save_game_data()
                        
                        # Lưu ván cờ vào PGN
                        self.save_game_to_pgn(winner)
                        
                        # Cập nhật thống kê
                        self.update_game_stats(winner)
                        
                        # Reset trò chơi
                        self.chess_board.reset()
                        self.current_turn = 'white'
                        self.selected_square = None
                        self.dragging = False
                        self.drag_piece = None
                        self.drag_pos = None
                        self.drag_from_pos = None
                        self.epoch += 1
                        self.game_logger.info(f"\nBắt đầu Epoch mới: {self.epoch}")
                    else:
                        # Kiểm tra chiếu tướng
                        if ChessRules.is_check(board_state, self.current_turn) and self.current_turn == 'black':
                            self.game_logger.info(f"Check! {self.current_turn} is in check!")
                        
                        # Chuyển lượt
                        self.current_turn = 'black' if self.current_turn == 'white' else 'white'
                        
                        # Nếu đang chơi với AI và đến lượt đen, thực hiện nước đi của AI
                        if self.current_turn == 'black' and self.game_mode == 'human_vs_ai':
                            self.game_logger.info("AI's turn")
                            self.make_ai_move()
                else:
                    if self.current_turn == 'black':
                        self.game_logger.info(f"Invalid move: {move}")
            except ValueError as e:
                if self.current_turn == 'black':
                    self.game_logger.error(f"Move error: {str(e)}")
            finally:
                # Reset trạng thái kéo thả
                self.dragging = False
                self.drag_piece = None
                self.drag_from_pos = None
                self.selected_square = None

    def handle_mouse_motion(self, pos):
        """
        Xử lý sự kiện di chuyển chuột.
        """
        if self.dragging and self.drag_piece:
            self.drag_pos = pos

    def run(self):
        """
        Chạy game loop.
        """
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    # Lưu thống kê cuối cùng
                    self.stats_logger.info("\n" + "="*50)
                    self.stats_logger.info("Kết thúc trò chơi")
                    self.stats_logger.info(f"Tổng số ván: {self.total_games}")
                    self.stats_logger.info(f"Thống kê cuối cùng:")
                    self.stats_logger.info(f"Trắng: {self.wins['white']}")
                    self.stats_logger.info(f"Đen: {self.wins['black']}")
                    self.stats_logger.info(f"Hòa: {self.wins['draws']}")
                    
                    # Lưu Elo rating cuối cùng
                    white_rating, black_rating = self.elo.get_ratings()
                    self.stats_logger.info(f"Elo Rating cuối cùng:")
                    self.stats_logger.info(f"Trắng: {white_rating}")
                    self.stats_logger.info(f"Đen: {black_rating}")
                    self.stats_logger.info("="*50 + "\n")
                    
                    # Lưu dữ liệu huấn luyện
                    self.data_collector.save_game_data()
                    
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left click
                        self.handle_click(event.pos)
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left click release
                        if self.dragging:
                            self.handle_click(event.pos)
                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Chuyển đổi chế độ chơi
                        self.game_mode = 'human_vs_ai' if self.game_mode == 'human_vs_human' else 'human_vs_human'
                        self.game_logger.info(f"\nChuyển chế độ chơi sang: {self.game_mode}")
                    elif event.key == pygame.K_r:
                        # Reset trò chơi
                        self.chess_board.reset()
                        self.current_turn = 'white'
                        self.selected_square = None
                        self.dragging = False
                        self.drag_piece = None
                        self.drag_pos = None
                        self.drag_from_pos = None
                        self.epoch += 1
                        self.game_logger.info(f"\nBắt đầu Epoch mới: {self.epoch}")
                    elif event.key == pygame.K_l:
                        # Load ván cờ từ PGN
                        pgn_files = [f for f in os.listdir(self.pgn_dir) if f.endswith('.pgn')]
                        if pgn_files:
                            latest_pgn = max(pgn_files, key=lambda x: os.path.getctime(os.path.join(self.pgn_dir, x)))
                            self.load_game_from_pgn(os.path.join(self.pgn_dir, latest_pgn))
                            self.game_logger.info(f"\nĐã load ván cờ từ file: {latest_pgn}")

        self.draw_board()
            pygame.display.flip()

        pygame.quit()
        sys.exit()

    def plot_game_stats(self, stats, filename='game_stats.png'):
        """Vẽ biểu đồ thống kê trò chơi."""
        plt.figure(figsize=(12, 6))
        
        # Biểu đồ tròn cho tỷ lệ thắng/thua/hòa
        plt.subplot(1, 2, 1)
        plt.pie(stats['wins'].values(), labels=stats['wins'].keys(), autopct='%1.1f%%')
        plt.title('Tỷ lệ kết quả trò chơi')
        
        # Biểu đồ đường cho Elo rating
        plt.subplot(1, 2, 2)
        if stats['elo_ratings']:
            white_ratings = [r[0] for r in stats['elo_ratings']]
            black_ratings = [r[1] for r in stats['elo_ratings']]
            plt.plot(white_ratings, label='Trắng')
            plt.plot(black_ratings, label='Đen')
            plt.title('Tiến triển Elo Rating')
            plt.xlabel('Số ván đấu')
            plt.ylabel('Elo Rating')
            plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_ai_learning(self, ai_data, filename='ai_learning.png'):
        """Vẽ biểu đồ quá trình học của AI."""
        plt.figure(figsize=(15, 10))
        
        # Biểu đồ epsilon
        plt.subplot(2, 2, 1)
        plt.plot(ai_data['epsilon'])
        plt.title('Tiến triển Epsilon')
        plt.xlabel('Số nước đi')
        plt.ylabel('Epsilon')
        
        # Biểu đồ phần thưởng
        plt.subplot(2, 2, 2)
        plt.plot(ai_data['rewards'])
        plt.title('Phần thưởng theo thời gian')
        plt.xlabel('Số nước đi')
        plt.ylabel('Phần thưởng')
        
        # Biểu đồ loss
        plt.subplot(2, 2, 3)
        plt.plot(ai_data['losses'])
        plt.title('Loss theo thời gian')
        plt.xlabel('Số nước đi')
        plt.ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_game_strategy(self, strategy_data, filename='game_strategy.png'):
        """Vẽ biểu đồ phân tích chiến lược."""
        plt.figure(figsize=(12, 6))
        
        # Biểu đồ cột cho các chỉ số chiến lược
        strategies = ['Kiểm soát trung tâm', 'Phát triển quân', 'Số lần chiếu', 'Giá trị quân bị ăn']
        values = [
            strategy_data['center_control'],
            strategy_data['piece_development'],
            strategy_data['check_count'],
            strategy_data['captures']
        ]
        
        plt.bar(strategies, values)
        plt.title('Phân tích chiến lược trò chơi')
        plt.ylabel('Số lần xuất hiện')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()
    
    def plot_ai_performance(self, performance_data, filename='ai_performance.png'):
        """Vẽ biểu đồ đánh giá hiệu suất AI."""
        plt.figure(figsize=(15, 10))
        
        # Biểu đồ tỷ lệ thắng
        plt.subplot(2, 2, 1)
        plt.plot(performance_data['win_rate'])
        plt.title('Tỷ lệ thắng của AI')
        plt.xlabel('Số phiên huấn luyện')
        plt.ylabel('Tỷ lệ thắng')
        
        # Biểu đồ phần thưởng trung bình
        plt.subplot(2, 2, 2)
        plt.plot(performance_data['avg_reward'])
        plt.title('Phần thưởng trung bình')
        plt.xlabel('Số phiên huấn luyện')
        plt.ylabel('Phần thưởng')
        
        # Biểu đồ loss trung bình
        plt.subplot(2, 2, 3)
        plt.plot(performance_data['avg_loss'])
        plt.title('Loss trung bình')
        plt.xlabel('Số phiên huấn luyện')
        plt.ylabel('Loss')
        
        # Biểu đồ epsilon
        plt.subplot(2, 2, 4)
        plt.plot(performance_data['epsilon'])
        plt.title('Giá trị Epsilon cuối cùng')
        plt.xlabel('Số phiên huấn luyện')
        plt.ylabel('Epsilon')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / filename)
        plt.close()

# Chạy thử nếu file được gọi trực tiếp
if __name__ == "__main__":
    board = ChessBoard()
    # Khởi tạo AI
    state_size = 64 * 12  # 64 ô, mỗi ô có 12 trạng thái có thể (6 loại quân * 2 màu)
    action_size = 64 * 64  # Số lượng nước đi có thể (từ ô này đến ô kia)
    q_network = QNetwork(state_size, action_size)
    agent = QLearningAgent(state_size, action_size)
    
    visualizer = ChessVisualizer(board, agent, q_network, "output")
    visualizer.run()
