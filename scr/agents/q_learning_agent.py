import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from scr.agents.experience_replay import ExperienceReplay
from scr.models.advanced_q_network import AdvancedQNetwork

class QLearningAgent:
    def __init__(self, state_size: int, action_size: int,
                 gamma: float = 0.99, epsilon: float = 1.0,
                 epsilon_min: float = 0.01, epsilon_decay: float = 0.995,
                 learning_rate: float = 0.001, replay_capacity: int = 10000,
                 batch_size: int = 64, target_update: int = 1000):
        """
        Khởi tạo QLearningAgent với các tham số:
        
        Args:
            state_size (int): Kích thước vector trạng thái
            action_size (int): Số lượng hành động có thể
            gamma (float): Hệ số chiết khấu
            epsilon (float): Tỷ lệ thăm dò ban đầu
            epsilon_min (float): Giá trị epsilon nhỏ nhất
            epsilon_decay (float): Tốc độ giảm epsilon
            learning_rate (float): Tốc độ học
            replay_capacity (int): Dung lượng bộ nhớ Experience Replay
            batch_size (int): Kích thước mẫu cho Experience Replay
            target_update (int): Tần suất cập nhật mạng đích
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        
        # Khởi tạo mạng chính và mạng đích
        self.policy_net = AdvancedQNetwork(state_size, action_size)
        self.target_net = AdvancedQNetwork(state_size, action_size)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Khởi tạo optimizer và loss function
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.policy_criterion = nn.CrossEntropyLoss()
        self.value_criterion = nn.MSELoss()
        
        # Khởi tạo bộ nhớ Experience Replay
        self.memory = ExperienceReplay(replay_capacity)
        
        # Biến theo dõi
        self.steps = 0
        self.loss_history = []
        
    def choose_action(self, state: torch.Tensor, legal_moves: list[int]) -> int:
        """
        Chọn hành động dựa trên chính sách ε-greedy.
        
        Args:
            state (torch.Tensor): Trạng thái hiện tại
            legal_moves (list[int]): Danh sách các hành động hợp lệ
            
        Returns:
            int: Hành động được chọn
        """
        return self.policy_net.get_action(state, legal_moves, self.epsilon)
    
    def store_transition(self, state: torch.Tensor, action: int,
                        reward: float, next_state: torch.Tensor, done: bool):
        """
        Lưu trữ transition vào bộ nhớ Experience Replay.
        
        Args:
            state (torch.Tensor): Trạng thái hiện tại
            action (int): Hành động đã thực hiện
            reward (float): Phần thưởng nhận được
            next_state (torch.Tensor): Trạng thái tiếp theo
            done (bool): Cờ cho biết ván chơi đã kết thúc
        """
        transition = (state, action, reward, next_state, done)
        self.memory.store_transition(transition)
    
    def experience_replay(self):
        """
        Lấy mẫu từ bộ nhớ và cập nhật mạng.
        """
        if len(self.memory) < self.batch_size:
            return
        
        # Lấy mẫu từ bộ nhớ
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Chuyển đổi sang tensor
        states = torch.cat(states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.cat(next_states)
        dones = torch.tensor(dones, dtype=torch.float32)
        
        # Tính Q-values hiện tại
        current_policy, current_value = self.policy_net(states)
        
        # Tính Q-values đích
        with torch.no_grad():
            next_policy, next_value = self.target_net(next_states)
            target_value = rewards + (1 - dones) * self.gamma * next_value.view(self.batch_size, -1).mean(dim=1)
        
        # Tính loss cho policy và value
        policy_loss = self.policy_criterion(current_policy, actions)
        value_loss = self.value_criterion(current_value.view(self.batch_size, -1).mean(dim=1), target_value)
        
        # Tổng loss
        loss = policy_loss + 0.5 * value_loss
        
        # Cập nhật mạng
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        # Lưu loss
        self.loss_history.append(loss.item())
        
        # Cập nhật mạng đích
        self.steps += 1
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Giảm epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def save_model(self, path: str):
        """
        Lưu model vào file.
        
        Args:
            path (str): Đường dẫn file
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'loss_history': self.loss_history
        }, path)
    
    def load_model(self, path: str):
        """
        Load model từ file.
        
        Args:
            path (str): Đường dẫn file
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.loss_history = checkpoint['loss_history']
    
    def get_loss_history(self) -> list[float]:
        """
        Lấy lịch sử loss.
        
        Returns:
            list[float]: Danh sách các giá trị loss
        """
        return self.loss_history
    
    def get_epsilon(self) -> float:
        """
        Lấy giá trị epsilon hiện tại.
        
        Returns:
            float: Giá trị epsilon
        """
        return self.epsilon
