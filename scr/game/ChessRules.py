class ChessRules:
    @staticmethod
    def is_check(board_state, current_turn):
        """
        Kiểm tra xem vua có đang bị chiếu không.
        
        Args:
            board_state (dict): Trạng thái bàn cờ.
            current_turn (str): Lượt hiện tại ('white' hoặc 'black').
            
        Returns:
            bool: True nếu vua đang bị chiếu, False nếu không.
        """
        # 1. Tìm vị trí vua
        king = 'K' if current_turn == 'white' else 'k'
        king_pos = None
        for pos, piece in board_state.items():
            if piece == king:
                king_pos = pos
                break
                
        if not king_pos:
            return False
            
        king_col = ord(king_pos[0]) - ord('a')
        king_row = int(king_pos[1])
        
        # 2. Kiểm tra chiếu từ quân tốt
        pawn_direction = -1 if current_turn == 'white' else 1
        for col_offset in [-1, 1]:
            col = king_col + col_offset
            row = king_row + pawn_direction
            if 0 <= col <= 7 and 1 <= row <= 8:
                pos = f"{chr(ord('a') + col)}{row}"
                piece = board_state.get(pos)
                if piece:
                    is_opponent_pawn = (piece == 'p' if current_turn == 'white' else piece == 'P')
                    if is_opponent_pawn:
                        return True
                        
        # 3. Kiểm tra chiếu từ quân mã
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        for col_offset, row_offset in knight_moves:
            col = king_col + col_offset
            row = king_row + row_offset
            if 0 <= col <= 7 and 1 <= row <= 8:
                pos = f"{chr(ord('a') + col)}{row}"
                piece = board_state.get(pos)
                if piece:
                    is_opponent_knight = (piece == 'n' if current_turn == 'white' else piece == 'N')
                    if is_opponent_knight:
                        return True
                        
        # 4. Kiểm tra chiếu theo đường thẳng (xe và hậu)
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        for col_dir, row_dir in directions:
            col = king_col + col_dir
            row = king_row + row_dir
            while 0 <= col <= 7 and 1 <= row <= 8:
                pos = f"{chr(ord('a') + col)}{row}"
                piece = board_state.get(pos)
                if piece:
                    is_opponent = (piece.islower() if current_turn == 'white' else piece.isupper())
                    if is_opponent:
                        piece_type = piece.upper()
                        if piece_type in ['R', 'Q']:
                            return True
                    break
                col += col_dir
                row += row_dir
                
        # 5. Kiểm tra chiếu theo đường chéo (tượng và hậu)
        diagonals = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
        for col_dir, row_dir in diagonals:
            col = king_col + col_dir
            row = king_row + row_dir
            while 0 <= col <= 7 and 1 <= row <= 8:
                pos = f"{chr(ord('a') + col)}{row}"
                piece = board_state.get(pos)
                if piece:
                    is_opponent = (piece.islower() if current_turn == 'white' else piece.isupper())
                    if is_opponent:
                        piece_type = piece.upper()
                        if piece_type in ['B', 'Q']:
                            return True
                    break
                col += col_dir
                row += row_dir
                
        # 6. Kiểm tra chiếu từ vua đối phương (để tránh vua đi cạnh vua)
        king_moves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        for col_offset, row_offset in king_moves:
            col = king_col + col_offset
            row = king_row + row_offset
            if 0 <= col <= 7 and 1 <= row <= 8:
                pos = f"{chr(ord('a') + col)}{row}"
                piece = board_state.get(pos)
                if piece:
                    is_opponent_king = (piece == 'k' if current_turn == 'white' else piece == 'K')
                    if is_opponent_king:
                        return True
                        
        return False
        
    @staticmethod
    def is_checkmate(board_state, current_turn):
        """
        Kiểm tra xem có phải là chiếu hết không.
        
        Args:
            board_state (dict): Trạng thái bàn cờ.
            current_turn (str): Lượt hiện tại ('white' hoặc 'black').
            
        Returns:
            bool: True nếu là chiếu hết, False nếu không.
        """
        # 1. Kiểm tra xem có đang bị chiếu không
        if not ChessRules.is_check(board_state, current_turn):
            return False
            
        # 2. Tìm vị trí vua
        king = 'K' if current_turn == 'white' else 'k'
        king_pos = None
        for pos, piece in board_state.items():
            if piece == king:
                king_pos = pos
                break
                
        if not king_pos:
            return False
            
        # 3. Thử tất cả các nước đi có thể của vua
        king_moves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        king_col = ord(king_pos[0]) - ord('a')
        king_row = int(king_pos[1])
        
        for col_offset, row_offset in king_moves:
            col = king_col + col_offset
            row = king_row + row_offset
            if 0 <= col <= 7 and 1 <= row <= 8:
                to_pos = f"{chr(ord('a') + col)}{row}"
                # Thử di chuyển vua
                if ChessRules.is_valid_move(board_state, f"{king_pos}{to_pos}", current_turn):
                    # Tạo bản sao trạng thái bàn cờ
                    new_board = board_state.copy()
                    # Di chuyển vua
                    new_board[to_pos] = new_board[king_pos]
                    del new_board[king_pos]
                    # Kiểm tra xem vua có còn bị chiếu không
                    if not ChessRules.is_check(new_board, current_turn):
                        return False
                        
        # 4. Thử tất cả các nước đi có thể của các quân khác
        for from_pos, piece in board_state.items():
            # Chỉ xét các quân cùng màu
            if current_turn == 'white' and not piece.isupper():
                continue
            if current_turn == 'black' and piece.isupper():
                continue
                
            # Bỏ qua vua vì đã xét ở trên
            if piece.upper() == 'K':
                continue
                
            # Thử di chuyển đến tất cả các ô trên bàn cờ
            for col in range(8):
                for row in range(1, 9):
                    to_pos = f"{chr(ord('a') + col)}{row}"
                    move = f"{from_pos}{to_pos}"
                    
                    # Kiểm tra nước đi có hợp lệ không
                    if ChessRules.is_valid_move(board_state, move, current_turn):
                        # Tạo bản sao trạng thái bàn cờ
                        new_board = board_state.copy()
                        # Thực hiện nước đi
                        new_board[to_pos] = new_board[from_pos]
                        del new_board[from_pos]
                        # Kiểm tra xem vua có còn bị chiếu không
                        if not ChessRules.is_check(new_board, current_turn):
                            return False
                            
        # Nếu không tìm thấy nước đi nào thoát chiếu => chiếu hết
        return True
        
    @staticmethod
    def is_valid_move(board_state, move, current_turn):
        """
        Kiểm tra nước đi có hợp lệ không.
        
        Args:
            board_state (dict): Trạng thái bàn cờ.
            move (str): Nước đi dạng 'e2e4'.
            current_turn (str): Lượt hiện tại ('white' hoặc 'black').
            
        Returns:
            bool: True nếu nước đi hợp lệ, False nếu không.
        """
        # 1. Kiểm tra định dạng nước đi
        if len(move) != 4:
            return False
            
        from_pos = move[:2]
        to_pos = move[2:]
        
        # 2. Kiểm tra vị trí nguồn và đích có hợp lệ không
        if not (ChessRules._is_valid_position(from_pos) and ChessRules._is_valid_position(to_pos)):
            return False
            
        # 3. Kiểm tra có quân cờ ở vị trí nguồn không
        if from_pos not in board_state:
            return False
            
        piece = board_state[from_pos]
        is_white_piece = piece.isupper()
        
        # 4. Kiểm tra lượt đi
        if (current_turn == 'white' and not is_white_piece) or \
           (current_turn == 'black' and is_white_piece):
            return False
            
        # 5. Kiểm tra không đi vào vị trí hiện tại
        if from_pos == to_pos:
            return False
            
        # 6. Kiểm tra không ăn quân cùng màu
        if to_pos in board_state:
            target_piece = board_state[to_pos]
            if target_piece.isupper() == is_white_piece:
                return False
                
        # 7. Kiểm tra luật di chuyển của quân cờ
        if not ChessRules._is_valid_piece_move(board_state, move, current_turn):
            return False
            
        # 8. Thử thực hiện nước đi và kiểm tra có bị chiếu không
        new_board = board_state.copy()
        new_board[to_pos] = new_board.pop(from_pos)
        
        # Nếu sau nước đi này vẫn bị chiếu, nước đi không hợp lệ
        if ChessRules.is_check(new_board, current_turn):
            return False
            
        return True
        
    @staticmethod
    def _is_valid_piece_move(board_state, move, current_turn):
        """
        Kiểm tra luật di chuyển của từng loại quân cờ.
        """
        from_pos = move[:2]
        to_pos = move[2:]
        piece = board_state[from_pos]
        piece_type = piece.upper()
        is_white = piece.isupper()
        
        if piece_type == 'P':
            return ChessRules._is_valid_pawn_move(board_state, from_pos, to_pos, is_white)
        elif piece_type == 'R':
            return ChessRules._is_valid_rook_move(board_state, from_pos, to_pos)
        elif piece_type == 'N':
            return ChessRules._is_valid_knight_move(board_state, from_pos, to_pos)
        elif piece_type == 'B':
            return ChessRules._is_valid_bishop_move(board_state, from_pos, to_pos)
        elif piece_type == 'Q':
            return ChessRules._is_valid_queen_move(board_state, from_pos, to_pos)
        elif piece_type == 'K':
            return ChessRules._is_valid_king_move(board_state, from_pos, to_pos)
            
        return False

    @staticmethod
    def _is_valid_position(pos):
        """
        Kiểm tra vị trí có nằm trong bàn cờ không.
        
        Args:
            pos (str): Vị trí cần kiểm tra (ví dụ: 'e4').
            
        Returns:
            bool: True nếu vị trí hợp lệ, False nếu không.
        """
        if len(pos) != 2:
            return False
            
        file = pos[0]  # Cột (a-h)
        rank = pos[1]  # Hàng (1-8)
        
        # Kiểm tra cột
        if not ('a' <= file <= 'h'):
            return False
            
        # Kiểm tra hàng
        try:
            rank_num = int(rank)
            if not (1 <= rank_num <= 8):
                return False
        except ValueError:
            return False
            
        return True

    @staticmethod
    def _is_valid_pawn_move(board_state, from_pos, to_pos, is_white):
        """
        Kiểm tra nước đi hợp lệ cho quân tốt.
        """
        from_col, from_row = ord(from_pos[0]) - ord('a'), int(from_pos[1])
        to_col, to_row = ord(to_pos[0]) - ord('a'), int(to_pos[1])
        
        direction = 1 if is_white else -1
        start_row = 2 if is_white else 7
        
        # Di chuyển thẳng
        if from_col == to_col:
            # Di chuyển 1 ô
            if to_row == from_row + direction:
                # Ô đích phải trống
                return board_state.get(to_pos) is None
            # Di chuyển 2 ô từ vị trí xuất phát
            if from_row == start_row and to_row == from_row + 2 * direction:
                intermediate_pos = f"{chr(ord('a') + from_col)}{from_row + direction}"
                # Cả ô trung gian và ô đích phải trống
                return (board_state.get(intermediate_pos) is None and 
                       board_state.get(to_pos) is None)
        # Ăn chéo
        elif abs(to_col - from_col) == 1 and to_row == from_row + direction:
            target_piece = board_state.get(to_pos)
            # Phải có quân đối phương ở ô đích
            return target_piece is not None and target_piece.isupper() != is_white
                
        return False

    @staticmethod
    def _is_valid_rook_move(board_state, from_pos, to_pos):
        """
        Kiểm tra nước đi hợp lệ cho quân xe.
        """
        from_col, from_row = ord(from_pos[0]) - ord('a'), int(from_pos[1])
        to_col, to_row = ord(to_pos[0]) - ord('a'), int(to_pos[1])
        
        # Xe chỉ di chuyển ngang hoặc dọc
        if from_col != to_col and from_row != to_row:
            return False
            
        # Kiểm tra không có quân cờ nào chặn đường
        if from_col == to_col:  # Di chuyển dọc
            step = 1 if to_row > from_row else -1
            for row in range(from_row + step, to_row, step):
                pos = f"{chr(ord('a') + from_col)}{row}"
                if board_state.get(pos):
                    return False
        else:  # Di chuyển ngang
            step = 1 if to_col > from_col else -1
            for col in range(from_col + step, to_col, step):
                pos = f"{chr(ord('a') + col)}{from_row}"
                if board_state.get(pos):
                    return False
                    
        return True

    @staticmethod
    def _is_valid_knight_move(board_state, from_pos, to_pos):
        """
        Kiểm tra nước đi hợp lệ cho quân mã.
        """
        from_col, from_row = ord(from_pos[0]) - ord('a'), int(from_pos[1])
        to_col, to_row = ord(to_pos[0]) - ord('a'), int(to_pos[1])
        
        col_diff = abs(to_col - from_col)
        row_diff = abs(to_row - from_row)
        
        # Mã di chuyển theo hình chữ L: 2 ô theo một hướng và 1 ô theo hướng vuông góc
        return (col_diff == 2 and row_diff == 1) or (col_diff == 1 and row_diff == 2)

    @staticmethod
    def _is_valid_bishop_move(board_state, from_pos, to_pos):
        """
        Kiểm tra nước đi hợp lệ cho quân tượng.
        """
        from_col, from_row = ord(from_pos[0]) - ord('a'), int(from_pos[1])
        to_col, to_row = ord(to_pos[0]) - ord('a'), int(to_pos[1])
        
        # Tượng chỉ di chuyển chéo
        if abs(to_col - from_col) != abs(to_row - from_row):
            return False
            
        # Kiểm tra không có quân cờ nào chặn đường
        col_step = 1 if to_col > from_col else -1
        row_step = 1 if to_row > from_row else -1
        
        col = from_col + col_step
        row = from_row + row_step
        
        while col != to_col:  # Chỉ cần kiểm tra một điều kiện vì đã xác nhận di chuyển chéo
            pos = f"{chr(ord('a') + col)}{row}"
            if board_state.get(pos) is not None:
                return False
            col += col_step
            row += row_step
            
        return True

    @staticmethod
    def _is_valid_queen_move(board_state, from_pos, to_pos):
        """
        Kiểm tra nước đi hợp lệ cho quân hậu.
        """
        # Hậu có thể di chuyển như xe hoặc tượng
        return (ChessRules._is_valid_rook_move(board_state, from_pos, to_pos) or
                ChessRules._is_valid_bishop_move(board_state, from_pos, to_pos))

    @staticmethod
    def _is_valid_king_move(board_state, from_pos, to_pos):
        """
        Kiểm tra nước đi hợp lệ cho quân vua.
        """
        from_col, from_row = ord(from_pos[0]) - ord('a'), int(from_pos[1])
        to_col, to_row = ord(to_pos[0]) - ord('a'), int(to_pos[1])
        
        col_diff = abs(to_col - from_col)
        row_diff = abs(to_row - from_row)
        
        # Vua chỉ có thể di chuyển 1 ô theo mọi hướng
        return col_diff <= 1 and row_diff <= 1 and not (col_diff == 0 and row_diff == 0) 