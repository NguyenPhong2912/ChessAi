import os
import pygame

class ChessPieceImages:
    def __init__(self):
        self.pieces = {}
        self.load_pieces()
    
    def load_pieces(self):
        """Tải hình ảnh các quân cờ"""
        # Đường dẫn đến thư mục chứa hình ảnh
        pieces_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'assets', 'images', 'imgs-80px')
        
        # Tên file cho từng quân cờ
        piece_files = {
            'P': 'white_pawn.png',
            'N': 'white_knight.png',
            'B': 'white_bishop.png',
            'R': 'white_rook.png',
            'Q': 'white_queen.png',
            'K': 'white_king.png',
            'p': 'black_pawn.png',
            'n': 'black_knight.png',
            'b': 'black_bishop.png',
            'r': 'black_rook.png',
            'q': 'black_queen.png',
            'k': 'black_king.png'
        }
        
        # Tải từng hình ảnh
        for piece_name, file_name in piece_files.items():
            file_path = os.path.join(pieces_dir, file_name)
            if os.path.exists(file_path):
                image = pygame.image.load(file_path)
                self.pieces[piece_name] = image
            else:
                print(f"Không tìm thấy file hình ảnh: {file_path}")
    
    def get_piece_image(self, piece_name):
        """Lấy hình ảnh của quân cờ"""
        return self.pieces.get(piece_name)
    
    def draw_piece(self, screen, piece_name, x, y, size):
        """Vẽ quân cờ lên màn hình"""
        image = self.get_piece_image(piece_name)
        if image:
            # Scale hình ảnh theo kích thước ô
            scaled_image = pygame.transform.scale(image, (size, size))
            screen.blit(scaled_image, (x, y)) 