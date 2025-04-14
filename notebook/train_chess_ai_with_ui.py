import os
import sys
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from pathlib import Path
import time
import json
import pygame
from pygame.locals import *
import chess
from collections import defaultdict

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
from scr.utils.chess_pieces import ChessPieceImages
from scr.agents.mcts import MCTS

# Khởi tạo Pygame
pygame.init()

# Các hằng số cho giao diện
WINDOW_SIZE = 1000  # Tăng kích thước cửa sổ để hiển thị thêm thông tin
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8
MARGIN = (WINDOW_SIZE - BOARD_SIZE) // 2

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (130, 151, 105)
BACKGROUND = (49, 46, 43)

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
    
    # Khởi tạo agent với mạng neural mới
    agent = QLearningAgent(
        state_size=input_size,
        action_size=output_size,
        learning_rate=0.0005,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.997,
        replay_capacity=50000,
        batch_size=128
    )
    
    # Khởi tạo trainer và evaluator
    trainer = Trainer(agent, engine)
    evaluator = ModelEvaluator(agent, engine)
    
    # Khởi tạo data collector
    data_collector = DataCollector()
    
    return engine, agent, trainer, evaluator, data_collector

def draw_board(screen, board):
    """Vẽ bàn cờ và các quân cờ"""
    # Vẽ nền
    screen.fill(BACKGROUND)
    
    # Vẽ bàn cờ
    for row in range(8):
        for col in range(8):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, 
                           (MARGIN + col * SQUARE_SIZE, 
                            MARGIN + row * SQUARE_SIZE, 
                            SQUARE_SIZE, SQUARE_SIZE))
    
    # Vẽ quân cờ
    piece_images = ChessPieceImages()
    for row in range(8):
        for col in range(8):
            # Chuyển đổi tọa độ sang ký hiệu cờ vua
            square = f"{chr(ord('a') + col)}{8 - row}"
            piece = board.piece_at(chess.parse_square(square))
            if piece:
                piece_images.draw_piece(screen, piece.symbol(), 
                                     MARGIN + col * SQUARE_SIZE,
                                     MARGIN + row * SQUARE_SIZE,
                                     SQUARE_SIZE)

def draw_stats(screen, episode, reward, win_rate, epsilon, moves, time_taken):
    """Vẽ thông tin thống kê"""
    font = pygame.font.Font(None, 32)
    y = 50
    stats = [
        f"Episode: {episode}",
        f"Reward: {reward:.2f}",
        f"Win Rate: {win_rate:.2%}",
        f"Epsilon: {epsilon:.4f}",
        f"Moves: {moves}",
        f"Time: {time_taken:.1f}s"
    ]
    
    for stat in stats:
        text = font.render(stat, True, WHITE)
        screen.blit(text, (BOARD_SIZE + MARGIN + 50, y))
        y += 40

def train_ai_with_ui(num_episodes=500, max_moves=200, max_time=600):
    """Huấn luyện AI với giao diện"""
    # Khởi tạo cửa sổ Pygame
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Chess AI Training")
    
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
        'checkmates': 0,
        'agent': agent
    }
    
    print("Bắt đầu huấn luyện AI...")
    
    # Thêm biến theo dõi tiến độ
    best_reward = float('-inf')
    no_improvement_count = 0
    
    for episode in range(num_episodes):
        # Đo thời gian bắt đầu episode
        episode_start = time.time()
        engine.reset()
        
        total_reward = 0
        moves = 0
        positions_seen = defaultdict(int)
        
        # Game loop
        running = True
        while running and moves < max_moves:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    return training_stats
            
            # Lấy trạng thái hiện tại
            state = engine.get_board_state_tensor()
            legal_moves = [move.from_square * 64 + move.to_square for move in engine.board.legal_moves]
            
            if not legal_moves:
                break
            
            # Sử dụng MCTS và ADMCDL để chọn nước đi
            if agent.epsilon < 0.3:
                mcts = MCTS(engine.board, num_simulations=200)
                move = mcts.get_best_move()
                action = move.from_square * 64 + move.to_square
            else:
                action = agent.choose_action(state, legal_moves)
            
            # Thực hiện nước đi
            from_square = action // 64
            to_square = action % 64
            move = chess.Move(from_square, to_square)
            
            if move in engine.board.legal_moves:
                # Kiểm tra lặp vị trí
                fen = engine.board.fen().split()[0]
                positions_seen[fen] += 1
                
                engine.board.push(move)
                
                # Tính toán phần thưởng
                reward = 0
                
                # Thưởng cho việc kiểm soát trung tâm
                central_squares = [27, 28, 35, 36]
                for square in central_squares:
                    piece = engine.board.piece_at(square)
                    if piece and piece.color == engine.board.turn:
                        reward += 0.1
                
                # Thưởng cho việc phát triển quân
                if moves < 10:
                    piece = engine.board.piece_at(to_square)
                    if piece and piece.piece_type in [chess.KNIGHT, chess.BISHOP]:
                        reward += 0.2
                
                # Phạt cho việc lặp vị trí
                if positions_seen[fen] > 2:
                    reward -= 0.5
                
                # Thưởng cho chiếu
                if engine.board.is_check():
                    reward += 0.3
                
                # Thưởng lớn cho chiếu hết
                if engine.board.is_checkmate():
                    reward = 100
                    training_stats['checkmates'] += 1
                elif engine.board.is_stalemate():
                    reward = -10
                
                total_reward += reward
                
                # Lưu trải nghiệm và huấn luyện
                next_state = engine.get_board_state_tensor()
                agent.store_transition(state, action, reward, next_state, engine.board.is_game_over())
                
                if len(agent.memory) > agent.batch_size:
                    agent.experience_replay()
                        
                # Cập nhật mạng ADMCDL
                agent.policy_net.update_network(state, action, reward, next_state, engine.board.is_game_over())
            
            moves += 1
            
            # Vẽ bàn cờ và thông tin
            draw_board(screen, engine.board)
            draw_stats(screen, episode + 1, total_reward, 
                      training_stats['checkmates'] / (episode + 1),
                      agent.epsilon, moves, time.time() - episode_start)
            pygame.display.flip()
            
            # Delay để có thể theo dõi
            pygame.time.wait(50)
            
            if engine.board.is_game_over():
                break
        
        # Cập nhật thống kê
        episode_time = time.time() - episode_start
        training_stats['episode_rewards'].append(total_reward)
        training_stats['epsilon_values'].append(agent.epsilon)
        training_stats['moves_per_episode'].append(moves)
        training_stats['time_per_episode'].append(episode_time)
        
        # Kiểm tra cải thiện
        if total_reward > best_reward:
            best_reward = total_reward
            no_improvement_count = 0
            # Lưu model tốt nhất
            best_model_path = os.path.join(root_dir, 'models', 'chess_ai_best.pth')
            agent.save_model(best_model_path)
        else:
            no_improvement_count += 1
        
        # Early stopping nếu không có cải thiện
        if no_improvement_count >= 50:
            print("\nDừng huấn luyện do không có cải thiện sau 50 episode")
            break
        
        # Tạo log chi tiết
        log_data = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'episode_stats': {
                'total_reward': total_reward,
                'checkmate_count': training_stats['checkmates'],
                'epsilon': agent.epsilon,
                'moves_count': moves,
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
        
        print(f"\nEpisode {episode + 1}/{num_episodes}:")
        print(f"Tổng phần thưởng: {total_reward:.2f}")
        print(f"Số ván chiếu hết: {training_stats['checkmates']}")
        print(f"Epsilon: {agent.epsilon:.4f}")
        print(f"Số nước đi: {moves}")
        print(f"Thời gian: {episode_time:.1f}s")
    
    return training_stats

def main():
    """Hàm chính"""
    # Tạo thư mục lưu model và logs nếu chưa tồn tại
    model_dir = os.path.join(root_dir, 'models')
    logs_dir = os.path.join(root_dir, 'logs')
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Huấn luyện AI với giao diện
    training_stats = train_ai_with_ui(
        num_episodes=500,    # Số episode huấn luyện
        max_moves=200,       # Số nước đi tối đa mỗi episode
        max_time=600         # Thời gian tối đa mỗi episode (giây)
    )
    
    # Lưu model
    final_model_path = os.path.join(model_dir, 'chess_ai_final.pth')
    training_stats['agent'].save_model(final_model_path)
    
    print(f"\nModel đã được lưu tại: {final_model_path}")
    
    # Đóng Pygame
    pygame.quit()

if __name__ == "__main__":
    main() 