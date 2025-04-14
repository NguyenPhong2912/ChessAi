import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from collections import defaultdict

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(residual)
        out = F.relu(out)
        return out

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

class MetaController(nn.Module):
    def __init__(self, state_size, num_options):
        super(MetaController, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_options)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class OptionNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(OptionNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, action_size)
        self.termination = nn.Linear(state_size, 1)
        
    def forward(self, x):
        features = F.relu(self.fc1(x))
        features = F.relu(self.fc2(features))
        return self.fc3(features), torch.sigmoid(self.termination(x))

class AdvancedQNetwork(nn.Module):
    def __init__(self, state_size, action_size, num_options=4):
        """
        Advanced Q-Network with convolutional layers and residual connections.
        
        Args:
            state_size (int): Size of the state vector (64 * 12 for chess)
            action_size (int): Number of possible actions (64 * 64 for chess)
        """
        super(AdvancedQNetwork, self).__init__()
        
        # Meta-controller để chọn option
        self.meta_controller = MetaController(state_size, num_options)
        
        # Các option network
        self.options = nn.ModuleList([
            OptionNetwork(state_size, action_size) 
            for _ in range(num_options)
        ])
        
        # Q-Network chính
        self.fc1 = nn.Linear(state_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, action_size)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(1024)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(256)
        
        # Dropout
        self.dropout = nn.Dropout(0.3)
        
        # MCTS parameters
        self.num_simulations = 200
        self.exploration_constant = 1.41
        
        # Lưu trữ thông tin training
        self.episode_rewards = []
        self.current_option = 0
        self.option_duration = 0
        
    def forward(self, x):
        # Thêm batch dimension nếu cần
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
            
        # Q-Network chính
        q_values = F.relu(self.bn1(self.fc1(x)))
        q_values = self.dropout(q_values)
        q_values = F.relu(self.bn2(self.fc2(q_values)))
        q_values = self.dropout(q_values)
        q_values = F.relu(self.bn3(self.fc3(q_values)))
        q_values = self.dropout(q_values)
        policy = self.fc4(q_values)
        
        # Meta-controller chọn option
        option_values = self.meta_controller(x)
        selected_option = torch.argmax(option_values, dim=1)[0] % len(self.options)
        
        # Thực thi option được chọn
        option_output, termination = self.options[selected_option](x)
        
        # Kết hợp output từ Q-Network và Option
        value = policy + 0.5 * option_output
        
        # Loại bỏ batch dimension nếu đã thêm
        if len(x.shape) == 2 and x.shape[0] == 1:
            policy = policy.squeeze(0)
            value = value.squeeze(0)
            
        return policy, value
        
    def get_action(self, state, legal_moves, epsilon=0.0):
        if not legal_moves:
            raise ValueError("No legal moves available")
            
        # Sử dụng MCTS khi không trong chế độ thăm dò
        if epsilon < 0.1 and len(legal_moves) > 1:
            from scr.agents.mcts import MCTS
            mcts = MCTS(state.board, num_simulations=self.num_simulations)
            best_move = mcts.get_best_move()
            return best_move.from_square * 64 + best_move.to_square
            
        # Epsilon-greedy với Q-values và options
        if np.random.random() < epsilon:
            return np.random.choice(legal_moves)
            
        self.eval()  # Chuyển sang chế độ eval
        with torch.no_grad():
            policy, value = self.forward(state)
            
            # Cập nhật option hiện tại
            self.option_duration += 1
            if self.option_duration >= 5:  # Đổi option sau mỗi 5 bước
                option_values = self.meta_controller(state)
                self.current_option = torch.argmax(option_values).item()
                self.option_duration = 0
            
            # Lấy output từ option hiện tại
            option_output, termination = self.options[self.current_option % len(self.options)](state)
            
            # Kết hợp Q-values và option output
            combined_q_values = value
            
            # Chỉ xem xét các nước đi hợp lệ
            legal_q_values = combined_q_values[legal_moves]
            self.train()  # Chuyển lại chế độ train
            return legal_moves[torch.argmax(legal_q_values).item()]
    
    def update_network(self, state, action, reward, next_state, done):
        """Cập nhật mạng neural sau mỗi bước"""
        self.eval()  # Chuyển sang chế độ eval
        # Cập nhật Q-Network
        with torch.no_grad():
            _, next_value = self.forward(next_state)
            max_next_q = torch.max(next_value)
            target = reward + (0.99 * max_next_q * (1 - done))
            target = torch.tensor([target], dtype=torch.float32)
        
        self.train()  # Chuyển lại chế độ train
        # Cập nhật meta-controller
        option_values = self.meta_controller(state)
        if len(option_values.shape) == 1:
            option_values = option_values.unsqueeze(0)
        option_target = target.expand(option_values.size(0), -1)
        option_loss = F.mse_loss(option_values[:, self.current_option:self.current_option+1], option_target)
        
        # Cập nhật option network
        option_output, termination = self.options[self.current_option % len(self.options)](state)
        
        # Xử lý action tensor
        if isinstance(action, int):
            action = torch.tensor([action])
        if len(option_output.shape) == 1:
            option_output = option_output.unsqueeze(0)
        action_value = option_output[0, action].unsqueeze(0)
        option_loss += F.mse_loss(action_value, target)
        
        # Lưu thông tin training
        self.episode_rewards.append(reward)
        
        return option_loss.item()

    def save_model(self, path):
        """Lưu model và các thông số"""
        torch.save({
            'model_state_dict': self.state_dict(),
            'meta_controller_state_dict': self.meta_controller.state_dict(),
            'options_state_dict': [opt.state_dict() for opt in self.options],
            'episode_rewards': self.episode_rewards,
            'current_option': self.current_option,
            'option_duration': self.option_duration
        }, path)
    
    def load_model(self, path):
        """Load model và các thông số"""
        checkpoint = torch.load(path)
        self.load_state_dict(checkpoint['model_state_dict'])
        self.meta_controller.load_state_dict(checkpoint['meta_controller_state_dict'])
        for opt, state_dict in zip(self.options, checkpoint['options_state_dict']):
            opt.load_state_dict(state_dict)
        self.episode_rewards = checkpoint['episode_rewards']
        self.current_option = checkpoint['current_option']
        self.option_duration = checkpoint['option_duration']

# Example usage
if __name__ == "__main__":
    # Create a sample state tensor (batch_size=1, state_size=64*12)
    state_size = 64 * 12
    action_size = 64 * 64
    state = torch.randn(1, state_size)
    
    # Create and test the network
    network = AdvancedQNetwork(state_size, action_size)
    policy, value = network(state)
    
    print(f"Policy output shape: {policy.shape}")
    print(f"Value output shape: {value.shape}")
    
    # Test action selection
    legal_moves = list(range(10))  # Example legal moves
    action = network.get_action(state, legal_moves, epsilon=0.1)
    print(f"Selected action: {action}")
    
    # Test position evaluation
    eval = network.evaluate_position(state)
    print(f"Position evaluation: {eval}") 