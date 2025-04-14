import os
import sys
import pygame
from pygame.locals import *
from pathlib import Path
import chess

# Xác định đường dẫn gốc và thêm vào Python path
try:
    root_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    root_dir = os.getcwd()

sys.path.insert(0, root_dir)

# Import các module từ dự án
from scr.game.ChessEngine import ChessEngine
from scr.agents.q_learning_agent import QLearningAgent
from scr.models.advanced_q_network import AdvancedQNetwork
from scr.utils.chess_pieces import ChessPieceImages

# Khởi tạo Pygame
pygame.init()

# Các hằng số cho giao diện
WINDOW_SIZE = 1200  # Tăng kích thước cửa sổ
BOARD_SIZE = 800    # Tăng kích thước bàn cờ
SQUARE_SIZE = BOARD_SIZE // 8
BOARD_MARGIN = (WINDOW_SIZE - BOARD_SIZE) // 4  # Căn giữa bàn cờ
INFO_PANEL_WIDTH = (WINDOW_SIZE - BOARD_SIZE) // 2  # Panel thông tin bên phải

# Màu sắc
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HIGHLIGHT = (130, 151, 105)
SUGGESTION = (155, 199, 0, 128)  # Màu gợi ý nước đi (có độ trong suốt)
BACKGROUND = (49, 46, 43)
TEXT_COLOR = (200, 200, 200)

# Thêm các hằng số cho hiệu ứng
CLICK_ANIMATION_DURATION = 100  # Thời gian hiệu ứng click (ms)
DRAG_ANIMATION_DURATION = 50    # Thời gian hiệu ứng kéo (ms)
CLICK_SCALE = 0.9              # Tỷ lệ thu nhỏ khi click
DRAG_SCALE = 1.1               # Tỷ lệ phóng to khi kéo

# Thêm biến trạng thái cho hiệu ứng
class GameState:
    def __init__(self):
        self.selected_square = None
        self.dragged_piece = None
        self.game_over = False
        self.current_player = 'white'
        self.message = ""
        self.click_start_time = 0
        self.drag_start_time = 0
        self.piece_scale = 1.0
        self.piece_offset = (0, 0)
        self.last_captured_piece = None  # Thêm biến theo dõi quân cờ cuối cùng bị ăn

def draw_move_suggestions(screen, board, selected_square):
    """Vẽ gợi ý các nước đi hợp lệ"""
    if selected_square:
        row, col = selected_square
        from_square = get_board_square(row, col)
        
        # Tạo surface trong suốt để vẽ gợi ý
        suggestion_surface = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(suggestion_surface, SUGGESTION, suggestion_surface.get_rect())
        
        # Hiển thị tất cả nước đi hợp lệ cho quân cờ được chọn
        for move in board.legal_moves:
            if move.from_square == from_square:
                to_row = 7 - (move.to_square // 8)
                to_col = move.to_square % 8
                screen.blit(suggestion_surface, 
                          (BOARD_MARGIN + to_col * SQUARE_SIZE,
                           BOARD_MARGIN + to_row * SQUARE_SIZE))

def draw_info_panel(screen, current_player, message, engine):
    """Vẽ panel thông tin bên phải"""
    panel_x = BOARD_MARGIN + BOARD_SIZE + 20
    y = BOARD_MARGIN
    
    # Vẽ nền cho panel
    panel_rect = pygame.Rect(panel_x, y, INFO_PANEL_WIDTH - 40, BOARD_SIZE)
    pygame.draw.rect(screen, (60, 60, 60), panel_rect)
    
    font = pygame.font.Font(None, 36)
    
    # Hiển thị lượt đi
    turn_text = f"Lượt: {'Trắng' if current_player == 'white' else 'Đen'}"
    text_surface = font.render(turn_text, True, TEXT_COLOR)
    screen.blit(text_surface, (panel_x + 20, y + 20))
    
    # Hiển thị thông báo
    if message:
        msg_surface = font.render(message, True, TEXT_COLOR)
        screen.blit(msg_surface, (panel_x + 20, y + 70))
    
    # Hiển thị số nước đi đã thực hiện
    moves_text = f"Số nước: {len(engine.board.move_stack)}"
    moves_surface = font.render(moves_text, True, TEXT_COLOR)
    screen.blit(moves_surface, (panel_x + 20, y + 120))

def draw_piece_with_animation(screen, piece, x, y, size, scale=1.0, offset=(0, 0)):
    """Vẽ quân cờ với hiệu ứng animation"""
    piece_images = ChessPieceImages()
    scaled_size = int(size * scale)
    x += offset[0]
    y += offset[1]
    piece_images.draw_piece(screen, piece.symbol(), x, y, scaled_size)

def draw_board(screen, board, game_state, mouse_pos=None):
    """Vẽ bàn cờ và các quân cờ với hiệu ứng"""
    # Vẽ nền
    screen.fill(BACKGROUND)
    
    # Vẽ bàn cờ
    for row in range(8):
        for col in range(8):
            color = LIGHT_SQUARE if (row + col) % 2 == 0 else DARK_SQUARE
            pygame.draw.rect(screen, color, 
                           (BOARD_MARGIN + col * SQUARE_SIZE, 
                            BOARD_MARGIN + row * SQUARE_SIZE, 
                            SQUARE_SIZE, SQUARE_SIZE))
    
    # Vẽ gợi ý nước đi
    draw_move_suggestions(screen, board, game_state.selected_square)
    
    # Vẽ quân cờ
    for row in range(8):
        for col in range(8):
            # Bỏ qua quân cờ đang được kéo
            if game_state.selected_square and (row, col) == game_state.selected_square and game_state.dragged_piece:
                continue
                
            square_idx = get_board_square(row, col)
            piece = board.piece_at(square_idx)
            if piece:
                x = BOARD_MARGIN + col * SQUARE_SIZE
                y = BOARD_MARGIN + row * SQUARE_SIZE
                
                # Tính toán hiệu ứng cho quân cờ được chọn
                scale = 1.0
                offset = (0, 0)
                
                if game_state.selected_square and (row, col) == game_state.selected_square:
                    if game_state.dragged_piece:
                        # Hiệu ứng kéo
                        elapsed = pygame.time.get_ticks() - game_state.drag_start_time
                        if elapsed < DRAG_ANIMATION_DURATION:
                            scale = DRAG_SCALE - (elapsed / DRAG_ANIMATION_DURATION) * (DRAG_SCALE - 1.0)
                    else:
                        # Hiệu ứng click
                        elapsed = pygame.time.get_ticks() - game_state.click_start_time
                        if elapsed < CLICK_ANIMATION_DURATION:
                            scale = CLICK_SCALE + (elapsed / CLICK_ANIMATION_DURATION) * (1.0 - CLICK_SCALE)
                
                draw_piece_with_animation(screen, piece, x, y, SQUARE_SIZE, scale, offset)
    
    # Vẽ ô được chọn
    if game_state.selected_square:
        row, col = game_state.selected_square
        pygame.draw.rect(screen, HIGHLIGHT,
                        (BOARD_MARGIN + col * SQUARE_SIZE,
                         BOARD_MARGIN + row * SQUARE_SIZE,
                         SQUARE_SIZE, SQUARE_SIZE), 3)
    
    # Vẽ quân cờ đang được kéo
    if game_state.dragged_piece and mouse_pos:
        x, y = mouse_pos
        draw_piece_with_animation(screen, game_state.dragged_piece,
                                x - SQUARE_SIZE // 2,
                                y - SQUARE_SIZE // 2,
                                SQUARE_SIZE, DRAG_SCALE)
    
    pygame.display.flip()

def get_square_from_pos(pos):
    """Chuyển đổi vị trí chuột thành tọa độ ô trên bàn cờ"""
    x, y = pos
    if BOARD_MARGIN <= x <= BOARD_MARGIN + BOARD_SIZE and BOARD_MARGIN <= y <= BOARD_MARGIN + BOARD_SIZE:
        col = (x - BOARD_MARGIN) // SQUARE_SIZE
        row = (y - BOARD_MARGIN) // SQUARE_SIZE
        return row, col
    return None

def get_board_square(row, col):
    """Chuyển đổi tọa độ (row, col) thành square index (0-63)"""
    # Đảo ngược row vì bàn cờ được vẽ từ trên xuống
    return (7 - row) * 8 + col

def draw_text(screen, text, pos, color=BLACK):
    """Vẽ text lên màn hình"""
    font = pygame.font.Font(None, 36)
    text_surface = font.render(text, True, color)
    screen.blit(text_surface, pos)

def handle_promotion(screen, square):
    """Xử lý phong tốt"""
    # Vẽ menu phong tốt
    menu_width = 200
    menu_height = 240
    menu_x = (WINDOW_SIZE - menu_width) // 2
    menu_y = (WINDOW_SIZE - menu_height) // 2
    
    menu_surface = pygame.Surface((menu_width, menu_height))
    menu_surface.fill((60, 60, 60))
    
    # Các quân có thể phong
    promotion_pieces = ['q', 'r', 'b', 'n']  # Hậu, xe, tượng, mã
    piece_images = ChessPieceImages()
    
    # Vẽ các lựa chọn
    for i, piece in enumerate(promotion_pieces):
        y = i * 60 + 10
        pygame.draw.rect(menu_surface, WHITE, (10, y, 180, 50))
        piece_images.draw_piece(menu_surface, piece, 70, y, 50)
    
    screen.blit(menu_surface, (menu_x, menu_y))
    pygame.display.flip()
    
    # Chờ người chơi chọn
    while True:
        for event in pygame.event.get():
            if event.type == MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                if menu_x <= mouse_pos[0] <= menu_x + menu_width:
                    relative_y = mouse_pos[1] - menu_y
                    if 0 <= relative_y <= menu_height:
                        piece_idx = relative_y // 60
                        if 0 <= piece_idx < len(promotion_pieces):
                            return promotion_pieces[piece_idx]
    return 'q'  # Mặc định là quân hậu

def is_promotion_move(board, from_square, to_square):
    """Kiểm tra xem có phải nước phong tốt không"""
    piece = board.piece_at(from_square)
    if piece is None or piece.piece_type != chess.PAWN:
        return False
    
    # Kiểm tra tốt trắng đến hàng 8 hoặc tốt đen đến hàng 1
    rank_8 = to_square >= 56  # Hàng 8 (từ a8 đến h8)
    rank_1 = to_square <= 7   # Hàng 1 (từ a1 đến h1)
    
    return (piece.color == chess.WHITE and rank_8) or (piece.color == chess.BLACK and rank_1)

def evaluate_position(board):
    """Đánh giá vị trí hiện tại"""
    if board.is_checkmate():
        return -1000 if board.turn else 1000
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
        
    # Giá trị quân cờ
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    # Bảng vị trí cho tốt (khuyến khích tiến lên và kiểm soát trung tâm)
    pawn_table = [
        0,  0,  0,  0,  0,  0,  0,  0,
        50, 50, 50, 50, 50, 50, 50, 50,
        10, 10, 20, 30, 30, 20, 10, 10,
        5,  5, 10, 25, 25, 10,  5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5, -5,-10,  0,  0,-10, -5,  5,
        5, 10, 10,-20,-20, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0
    ]
    
    score = 0
    # Tính điểm dựa trên vật chất và vị trí
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            value = piece_values[piece.piece_type]
            # Cộng thêm điểm vị trí cho tốt
            if piece.piece_type == chess.PAWN:
                pos_value = pawn_table[square if piece.color else 63 - square]
                value += pos_value
                
            score += value if piece.color == chess.WHITE else -value
            
    # Thêm điểm cho kiểm soát trung tâm
    central_squares = [27, 28, 35, 36]
    for square in central_squares:
        piece = board.piece_at(square)
        if piece:
            score += 30 if piece.color == chess.WHITE else -30
            
    # Thêm điểm cho tính cơ động
    score += len(list(board.legal_moves)) * 10
    
    return score

def get_best_move(board, depth=3):
    """Tìm nước đi tốt nhất sử dụng minimax với alpha-beta pruning"""
    def minimax(board, depth, alpha, beta, maximizing_player):
        if depth == 0 or board.is_game_over():
            return evaluate_position(board)
            
        if maximizing_player:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    best_move = None
    best_value = float('-inf')
    alpha = float('-inf')
    beta = float('inf')
    
    for move in board.legal_moves:
        board.push(move)
        value = minimax(board, depth - 1, alpha, beta, False)
        board.pop()
        if value > best_value:
            best_value = value
            best_move = move
            
    return best_move

def only_kings_left(board):
    """Kiểm tra xem trên bàn cờ chỉ còn lại quân vua không"""
    pieces = 0
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            if piece.piece_type != chess.KING:
                return False
            pieces += 1
    return pieces == 2  # Chỉ còn 2 quân vua

def main():
    """Hàm chính"""
    # Khởi tạo cửa sổ Pygame
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("Chess Game")
    
    # Khởi tạo engine
    engine = ChessEngine()
    
    # Khởi tạo advanced network với ADMCDL
    network = AdvancedQNetwork(
        state_size=64 * 12,
        action_size=64 * 64,
        num_options=4
    )
    
    # Khởi tạo agent
    agent = QLearningAgent(
        state_size=64 * 12,
        action_size=64 * 64,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        learning_rate=0.0005,
        replay_capacity=50000,
        batch_size=128,
        target_update=1000
    )
    
    # Load model đã huấn luyện
    model_path = os.path.join(root_dir, 'models', 'chess_ai_best.pth')
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print("Đã load model thành công!")
    else:
        print("Không tìm thấy model đã huấn luyện!")
        print("Vui lòng chạy script huấn luyện trước:")
        print("python3 notebook/train_chess_ai_with_ui.py")
        return
    
    # Khởi tạo trạng thái game
    game_state = GameState()
    
    # Game loop
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        current_time = pygame.time.get_ticks()
        
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
                
            elif event.type == MOUSEBUTTONDOWN and not game_state.game_over:
                if event.button == 1:  # Click chuột trái
                    square = get_square_from_pos(event.pos)
                    if square:
                        row, col = square
                        square_idx = get_board_square(row, col)
                        piece = engine.board.piece_at(square_idx)
                        
                        if game_state.selected_square is None:
                            # Chọn quân cờ
                            if piece:
                                if (piece.color == chess.WHITE and game_state.current_player == 'white') or \
                                   (piece.color == chess.BLACK and game_state.current_player == 'black'):
                                    game_state.selected_square = (row, col)
                                    game_state.dragged_piece = piece
                                    game_state.click_start_time = current_time
                        
            elif event.type == MOUSEBUTTONUP and not game_state.game_over:
                if event.button == 1 and game_state.selected_square and game_state.dragged_piece:
                    game_state.drag_start_time = current_time
                    square = get_square_from_pos(event.pos)
                    if square:
                        from_row, from_col = game_state.selected_square
                        to_row, to_col = square
                        
                        # Tính toán nước đi
                        from_square = get_board_square(from_row, from_col)
                        to_square = get_board_square(to_row, to_col)
                        
                        # Kiểm tra phong tốt
                        if is_promotion_move(engine.board, from_square, to_square):
                            promotion_piece = handle_promotion(screen, to_square)
                            move = chess.Move(from_square, to_square, promotion=chess.Piece.from_symbol(promotion_piece).piece_type)
                        else:
                            move = chess.Move(from_square, to_square)
                        
                        if move in engine.board.legal_moves:
                            # Lưu quân cờ bị ăn trước khi thực hiện nước đi
                            captured_piece = engine.board.piece_at(to_square)
                            if captured_piece:
                                game_state.last_captured_piece = captured_piece
                            
                            engine.board.push(move)
                            game_state.message = ""
                            
                            # Chuyển lượt cho AI
                            game_state.current_player = 'black'
                            
                            # AI thực hiện nước đi
                            state = engine.get_board_state_tensor()
                            legal_moves = [m.from_square * 64 + m.to_square for m in engine.board.legal_moves]
                            
                            if legal_moves:
                                action = agent.choose_action(state, legal_moves)
                                from_square = action // 64
                                to_square = action % 64
                                
                                # Lưu quân cờ bị ăn bởi AI
                                captured_piece = engine.board.piece_at(to_square)
                                if captured_piece:
                                    game_state.last_captured_piece = captured_piece
                                
                                if is_promotion_move(engine.board, from_square, to_square):
                                    move = chess.Move(from_square, to_square, promotion=chess.QUEEN)
                                else:
                                    move = chess.Move(from_square, to_square)
                                    
                                engine.board.push(move)
                            
                            game_state.current_player = 'white'
                            
                            # Kiểm tra kết thúc game
                            if engine.board.is_checkmate():
                                game_state.message = "Chiếu hết!"
                                game_state.game_over = True
                            elif engine.board.is_stalemate():
                                game_state.message = "Hòa do bế tắc!"
                                game_state.game_over = True
                            elif only_kings_left(engine.board):
                                if game_state.last_captured_piece:
                                    winner = "Đen" if game_state.last_captured_piece.color == chess.WHITE else "Trắng"
                                    game_state.message = f"Trò chơi kết thúc - {winner} thắng!"
                                else:
                                    game_state.message = "Trò chơi kết thúc - Hòa!"
                                game_state.game_over = True
                            elif len(list(engine.board.legal_moves)) == 0:
                                game_state.message = "Hòa do không còn nước đi hợp lệ!"
                                game_state.game_over = True
                        else:
                            game_state.message = "Nước đi không hợp lệ!"
                    
                    game_state.selected_square = None
                    game_state.dragged_piece = None
        
        # Vẽ bàn cờ
        draw_board(screen, engine.board, game_state, mouse_pos)
        
        # Vẽ thông tin game
        draw_info_panel(screen, game_state.current_player, game_state.message, engine)
    
    pygame.quit()

if __name__ == "__main__":
    main()