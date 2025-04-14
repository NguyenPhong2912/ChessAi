from scr.game.ChessRules import ChessRules

class ChessBoard:
    """
    Lớp ChessBoard mô phỏng bàn cờ vua, hỗ trợ trạng thái hiện tại của ván đấu, 
    cập nhật nước đi, kiểm tra ván cờ đã kết thúc hay chưa và đặt lại bàn cờ.
    """
    
    def __init__(self):
        """
        Khởi tạo bàn cờ vua với vị trí ban đầu.
        """
        self.board = {}
        self.current_turn = 'white'
        self.reset()

    def reset(self):
        """
        Reset bàn cờ về vị trí ban đầu.
        """
        # Xóa bàn cờ
        self.board.clear()
        
        # Đặt quân trắng
        self.board['a1'] = 'R'
        self.board['b1'] = 'N'
        self.board['c1'] = 'B'
        self.board['d1'] = 'Q'
        self.board['e1'] = 'K'
        self.board['f1'] = 'B'
        self.board['g1'] = 'N'
        self.board['h1'] = 'R'
        for i in range(8):
            self.board[f"{chr(ord('a') + i)}2"] = 'P'
            
        # Đặt quân đen
        self.board['a8'] = 'r'
        self.board['b8'] = 'n'
        self.board['c8'] = 'b'
        self.board['d8'] = 'q'
        self.board['e8'] = 'k'
        self.board['f8'] = 'b'
        self.board['g8'] = 'n'
        self.board['h8'] = 'r'
        for i in range(8):
            self.board[f"{chr(ord('a') + i)}7"] = 'p'
            
        self.current_turn = 'white'

    def get_state(self):
        """
        Lấy trạng thái hiện tại của bàn cờ.
        
        Returns:
            dict: Trạng thái bàn cờ
        """
        return self.board.copy()

    def apply_move(self, move):
        """
        Áp dụng nước đi lên bàn cờ.
        
        Args:
            move (str): Nước đi dạng 'e2e4' hoặc 'e7e8q' (cho phong cấp)
            
        Raises:
            ValueError: Nếu nước đi không hợp lệ
        """
        if not ChessRules.is_valid_move(self.board, move, self.current_turn):
            raise ValueError("Nước đi không hợp lệ")
            
        from_pos = move[:2]
        to_pos = move[2:4]  # Lấy 2 ký tự đầu của phần đích
        promotion = move[4] if len(move) > 4 else None  # Ký tự phong cấp nếu có
        
        # Di chuyển quân cờ
        piece = self.board[from_pos]
        
        # Xử lý phong cấp cho quân lính
        if piece.upper() == 'P':
            # Kiểm tra quân lính đã đến hàng cuối chưa
            if (piece.isupper() and to_pos[1] == '8') or (piece.islower() and to_pos[1] == '1'):
                if not promotion:
                    raise ValueError("Cần chỉ định quân phong cấp (q, r, b, n)")
                # Chuyển đổi ký tự phong cấp thành quân cờ
                promotion_pieces = {
                    'q': 'Q' if piece.isupper() else 'q',
                    'r': 'R' if piece.isupper() else 'r',
                    'b': 'B' if piece.isupper() else 'b',
                    'n': 'N' if piece.isupper() else 'n'
                }
                if promotion not in promotion_pieces:
                    raise ValueError("Quân phong cấp không hợp lệ (chỉ được q, r, b, n)")
                self.board[to_pos] = promotion_pieces[promotion]
            else:
                self.board[to_pos] = piece
        else:
            self.board[to_pos] = piece
            
        del self.board[from_pos]
        
        # Kiểm tra chiếu hết sau nước đi
        next_turn = 'black' if self.current_turn == 'white' else 'white'
        if ChessRules.is_checkmate(self.board, next_turn):
            self.current_turn = next_turn  # Giữ nguyên lượt của người chiếu hết
            return
            
        # Đổi lượt nếu không phải chiếu hết
        self.current_turn = next_turn

    def get_legal_moves(self):
        """
        Lấy danh sách các nước đi hợp lệ.
        
        Returns:
            list: Danh sách các nước đi hợp lệ
        """
        legal_moves = []
        board_state = self.get_state()
        
        # Duyệt qua tất cả các vị trí trên bàn cờ
        for from_pos in board_state:
            piece = board_state[from_pos]
            # Chỉ xét quân cờ của lượt hiện tại
            if (piece.isupper() and self.current_turn == 'white') or \
               (piece.islower() and self.current_turn == 'black'):
                # Duyệt qua tất cả các vị trí đích có thể
                for to_pos in board_state:
                    move = f"{from_pos}{to_pos}"
                    if ChessRules.is_valid_move(board_state, move, self.current_turn):
                        legal_moves.append(move)
                        
        return legal_moves

    def is_check(self):
        """
        Kiểm tra xem vua có đang bị chiếu không.
        
        Returns:
            bool: True nếu vua bị chiếu, False nếu không
        """
        # Tìm vị trí vua
        king_pos = None
        for pos, piece in self.board.items():
            if (piece == 'K' and self.current_turn == 'white') or \
               (piece == 'k' and self.current_turn == 'black'):
                king_pos = pos
                break
                
        if not king_pos:
            return False
            
        # Kiểm tra xem có quân nào có thể tấn công vua không
        board_state = self.get_state()
        for from_pos in board_state:
            piece = board_state[from_pos]
            # Chỉ xét quân của đối phương
            if (piece.isupper() and self.current_turn == 'black') or \
               (piece.islower() and self.current_turn == 'white'):
                move = f"{from_pos}{king_pos}"
                if ChessRules.is_valid_move(board_state, move, 'black' if self.current_turn == 'white' else 'white'):
                    return True
                    
        return False

    def is_checkmate(self):
        """
        Kiểm tra xem có phải là chiếu hết không.
        
        Returns:
            bool: True nếu là chiếu hết, False nếu không
        """
        if not self.is_check():
            return False
            
        # Nếu đang bị chiếu và không có nước đi hợp lệ nào
        return len(self.get_legal_moves()) == 0


# Ví dụ sử dụng:
if __name__ == "__main__":
    board = ChessBoard()
    print("Trạng thái ban đầu:", board.get_state())
    
    try:
        board.apply_move("e2e4")
        print("Trạng thái sau khi di chuyển:", board.get_state())
    except ValueError as e:
        print(e)
    
    print("Ván cờ đã kết thúc?", board.is_game_over())

    board.reset()
    print("Trạng thái sau khi reset:", board.get_state())
