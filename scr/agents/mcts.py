import chess
import numpy as np
import math

class Node:
    def __init__(self, board, parent=None, move=None):
        self.board = board
        self.parent = parent
        self.move = move
        self.children = []
        self.wins = 0
        self.visits = 0
        self.untried_moves = list(board.legal_moves)
        
    def ucb_score(self, exploration=1.41):
        if self.visits == 0:
            return float('inf')
        return (self.wins / self.visits) + exploration * math.sqrt(math.log(self.parent.visits) / self.visits)
    
    def expand(self):
        move = self.untried_moves.pop()
        new_board = self.board.copy()
        new_board.push(move)
        child = Node(new_board, parent=self, move=move)
        self.children.append(child)
        return child
        
    def is_terminal(self):
        return self.board.is_game_over()
        
    def rollout(self):
        current_board = self.board.copy()
        while not current_board.is_game_over():
            moves = list(current_board.legal_moves)
            if not moves:
                break
            move = np.random.choice(moves)
            current_board.push(move)
        return self.get_result(current_board)
        
    def get_result(self, board):
        if board.is_checkmate():
            return 1 if board.turn else -1
        return 0
        
    def backpropagate(self, result):
        self.visits += 1
        self.wins += result
        if self.parent:
            self.parent.backpropagate(-result)

class MCTS:
    def __init__(self, board, num_simulations=200):
        self.root = Node(board)
        self.num_simulations = num_simulations
        
    def get_best_move(self):
        for _ in range(self.num_simulations):
            node = self.select()
            if not node.is_terminal():
                node = node.expand()
                result = node.rollout()
                node.backpropagate(result)
        
        # Chọn nước đi tốt nhất dựa trên số lần thăm
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.move
        
    def select(self):
        node = self.root
        while node.untried_moves == [] and node.children != []:
            node = max(node.children, key=lambda c: c.ucb_score())
        return node 