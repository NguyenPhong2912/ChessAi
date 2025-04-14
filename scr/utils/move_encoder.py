import chess

class MoveEncoder:
    """
    A class to encode and decode chess moves in UCI notation to/from integer indices.
    """

    def __init__(self):
        # Generate mapping for all possible moves.
        # Note: This is a basic mapping and may not cover all intricacies of move legality.
        self.move_to_index = {}
        self.index_to_move = {}
        self._generate_move_mapping()

    def _generate_move_mapping(self):
        """
        Generate a mapping for all possible moves (and promotions) in UCI format.
        This mapping covers moves from every source square to every target square,
        including basic promotion moves.
        """
        squares = [chess.square_name(sq) for sq in chess.SQUARES]
        promotions = ['q', 'r', 'b', 'n']
        index = 0

        for source in squares:
            for target in squares:
                move_str = source + target
                # Basic move mapping
                self.move_to_index[move_str] = index
                self.index_to_move[index] = move_str
                index += 1

                # Add promotion moves if the move is a pawn promotion candidate.
                # For white: pawn from rank 7 to rank 8.
                # For black: pawn from rank 2 to rank 1.
                if (source[1] == '7' and target[1] == '8') or (source[1] == '2' and target[1] == '1'):
                    for promo in promotions:
                        move_str_promo = source + target + promo
                        self.move_to_index[move_str_promo] = index
                        self.index_to_move[index] = move_str_promo
                        index += 1

    def encode(self, move: str) -> int:
        """
        Encode a move in UCI notation to an integer index.

        Args:
            move (str): Move in UCI notation (e.g., "e2e4" or "e7e8q").

        Returns:
            int: The corresponding integer index, or -1 if not found.
        """
        return self.move_to_index.get(move, -1)

    def decode(self, index: int) -> str:
        """
        Decode an integer index back to a move in UCI notation.

        Args:
            index (int): The integer index.

        Returns:
            str: The corresponding move in UCI notation, or None if not found.
        """
        return self.index_to_move.get(index, None)
