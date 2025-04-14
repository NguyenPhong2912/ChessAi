import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from scr.data.DataCollector import DataCollector
from scr.models.q_network import QNetwork

class ChessDataset(Dataset):
    def __init__(self, states, actions, rewards):
        self.states = torch.FloatTensor(states)
        self.actions = torch.LongTensor(actions)
        self.rewards = torch.FloatTensor(rewards)
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.rewards[idx]

class ModelTrainer:
    def __init__(self, state_size=768, action_size=4096, hidden_dim=256,
                 learning_rate=0.001, batch_size=32):
        """
        Khởi tạo ModelTrainer.
        
        Args:
            state_size (int): Kích thước vector trạng thái (64 ô * 12 loại quân).
            action_size (int): Số lượng hành động có thể (64 ô nguồn * 64 ô đích).
            hidden_dim (int): Số nơ-ron trong các lớp ẩn.
            learning_rate (float): Tốc độ học.
            batch_size (int): Kích thước batch khi huấn luyện.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.q_network = QNetwork(state_size, action_size, hidden_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        self.batch_size = batch_size
        
    def prepare_data(self, data_dir="data/training"):
        """
        Chuẩn bị dữ liệu huấn luyện từ các file đã lưu.
        
        Args:
            data_dir (str): Thư mục chứa dữ liệu huấn luyện.
            
        Returns:
            ChessDataset: Dataset chứa dữ liệu huấn luyện.
        """
        # Đọc dữ liệu
        states, moves, rewards, _ = DataCollector.load_training_data(data_dir)
        
        # Chuyển đổi moves thành indices
        action_indices = []
        for move in moves:
            from_pos = move[:2]
            to_pos = move[2:]
            from_idx = (ord(from_pos[0]) - ord('a') + (int(from_pos[1]) - 1) * 8)
            to_idx = (ord(to_pos[0]) - ord('a') + (int(to_pos[1]) - 1) * 8)
            action_idx = from_idx * 64 + to_idx
            action_indices.append(action_idx)
            
        return ChessDataset(states, action_indices, rewards)
        
    def train(self, num_epochs=100, data_dir="data/training", save_path="models/q_network.pth"):
        """
        Huấn luyện mô hình.
        
        Args:
            num_epochs (int): Số epoch huấn luyện.
            data_dir (str): Thư mục chứa dữ liệu huấn luyện.
            save_path (str): Đường dẫn lưu mô hình.
        """
        # Chuẩn bị dữ liệu
        dataset = self.prepare_data(data_dir)
        if len(dataset) == 0:
            print("Không có dữ liệu huấn luyện!")
            return
            
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        print(f"Bắt đầu huấn luyện trên {len(dataset)} mẫu...")
        
        # Huấn luyện
        for epoch in range(num_epochs):
            total_loss = 0
            for states, actions, rewards in dataloader:
                # Chuyển dữ liệu sang device
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                
                # Forward pass
                q_values = self.q_network(states)
                predicted_rewards = q_values.gather(1, actions.unsqueeze(1)).squeeze()
                
                # Tính loss và cập nhật
                loss = self.criterion(predicted_rewards, rewards)
                total_loss += loss.item()
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            # In thông tin sau mỗi epoch
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
            
        # Lưu mô hình
        torch.save(self.q_network.state_dict(), save_path)
        print(f"Đã lưu mô hình tại {save_path}")
        
    def load_model(self, model_path):
        """
        Tải mô hình đã huấn luyện.
        
        Args:
            model_path (str): Đường dẫn đến file mô hình.
        """
        self.q_network.load_state_dict(torch.load(model_path))
        self.q_network.eval() 