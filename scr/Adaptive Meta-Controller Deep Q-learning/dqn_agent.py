import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List
from collections import deque
import random

class DQN(nn.Module):
    def __init__(self, input_channels: int, board_size: int, action_size: int):
        """
        Khởi tạo mạng DQN
        
        Args:
            input_channels (int): Số kênh đầu vào (12 cho cờ vua)
            board_size (int): Kích thước bàn cờ (8)
            action_size (int): Số lượng hành động có thể (4096)
        """
        super(DQN, self).__init__()
        
        # Các lớp CNN
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        # Các lớp fully connected
        self.fc1 = nn.Linear(256 * board_size * board_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_size)
        
        # Dropout để tránh overfitting
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor shape (batch_size, channels, height, width)
            
        Returns:
            torch.Tensor: Q-values cho mỗi hành động
        """
        # CNN layers
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

class DQNAgent:
    def __init__(self, state_channels: int, board_size: int, action_size: int,
                 learning_rate: float = 0.001, gamma: float = 0.99,
                 epsilon: float = 1.0, epsilon_min: float = 0.01,
                 epsilon_decay: float = 0.995, memory_size: int = 10000,
                 batch_size: int = 64, target_update: int = 1000):
        """
        Khởi tạo DQN Agent
        
        Args:
            state_channels (int): Số kênh trạng thái
            board_size (int): Kích thước bàn cờ
            action_size (int): Số lượng hành động
            learning_rate (float): Tốc độ học
            gamma (float): Hệ số chiết khấu
            epsilon (float): Tỷ lệ thăm dò ban đầu
            epsilon_min (float): Giá trị epsilon nhỏ nhất
            epsilon_decay (float): Tốc độ giảm epsilon
            memory_size (int): Dung lượng bộ nhớ
            batch_size (int): Kích thước batch
            target_update (int): Tần suất cập nhật mạng đích
        """
        self.state_channels = state_channels
        self.board_size = board_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Khởi tạo mạng
        self.policy_net = DQN(state_channels, board_size, action_size)
        self.target_net = DQN(state_channels, board_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer và loss function
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Bộ nhớ replay
        self.memory = deque(maxlen=memory_size)
        
        # Theo dõi số bước
        self.steps = 0
        
    def remember(self, state: torch.Tensor, action: int, reward: float,
                next_state: torch.Tensor, done: bool):
        """
        Lưu trữ transition vào bộ nhớ
        
        Args:
            state (torch.Tensor): Trạng thái hiện tại
            action (int): Hành động đã thực hiện
            reward (float): Phần thưởng
            next_state (torch.Tensor): Trạng thái tiếp theo
            done (bool): Cờ kết thúc
        """
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state: torch.Tensor, legal_moves: List[int]) -> int:
        """
        Chọn hành động theo chính sách ε-greedy
        
        Args:
            state (torch.Tensor): Trạng thái hiện tại
            legal_moves (List[int]): Danh sách các nước đi hợp lệ
            
        Returns:
            int: Hành động được chọn
        """
        if random.random() < self.epsilon:
            return random.choice(legal_moves)
            
        with torch.no_grad():
            state = state.unsqueeze(0)
            q_values = self.policy_net(state)
            
            # Chỉ xét các nước đi hợp lệ
            mask = torch.zeros_like(q_values)
            mask[0, legal_moves] = 1
            q_values = q_values * mask
            
            return q_values.argmax().item()
            
    def replay(self) -> float:
        """
        Học từ bộ nhớ replay
        
        Returns:
            float: Loss trung bình
        """
        if len(self.memory) < self.batch_size:
            return 0.0
            
        # Lấy mẫu từ bộ nhớ
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Chuyển đổi sang tensor
        states = torch.cat(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Tính Q-values hiện tại
        current_q_values = self.policy_net(states)
        current_q_values = current_q_values.gather(1, actions.unsqueeze(1))
        
        # Tính Q-values đích
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
            
        # Tính loss và cập nhật mạng
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Cập nhật mạng đích
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
            
        # Giảm epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        return loss.item()
        
    def save(self, path: str):
        """
        Lưu model
        
        Args:
            path (str): Đường dẫn file
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps
        }, path)
        
    def load(self, path: str):
        """
        Load model
        
        Args:
            path (str): Đường dẫn file
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps'] 