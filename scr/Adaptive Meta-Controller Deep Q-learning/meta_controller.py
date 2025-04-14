import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple

class MetaController(nn.Module):
    def __init__(self, input_size: int = 10, hidden_size: int = 64):
        """
        Khởi tạo Meta-Controller
        
        Args:
            input_size (int): Kích thước vector đầu vào (số metrics)
            hidden_size (int): Kích thước lớp ẩn
        """
        super(MetaController, self).__init__()
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 3)  # 3 hệ số điều chỉnh
        
        # Dropout để tránh overfitting
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Vector metrics đầu vào
            
        Returns:
            torch.Tensor: 3 hệ số điều chỉnh (epsilon_decay, learning_rate, gamma)
        """
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))  # Đảm bảo output trong [0,1]
        
        # Chuyển đổi sang khoảng [0.5, 1.5]
        return 0.5 + x
        
    def get_adjustment_factors(self, metrics: List[float]) -> Tuple[float, float, float]:
        """
        Lấy các hệ số điều chỉnh từ metrics
        
        Args:
            metrics (List[float]): Danh sách các metrics
            
        Returns:
            Tuple[float, float, float]: (epsilon_decay_factor, learning_rate_factor, gamma_factor)
        """
        with torch.no_grad():
            x = torch.tensor(metrics, dtype=torch.float32)
            factors = self(x)
            return tuple(factors.numpy())

class MetaControllerTrainer:
    def __init__(self, input_size: int = 10, hidden_size: int = 64,
                 learning_rate: float = 0.0001):
        """
        Khởi tạo Meta-Controller Trainer
        
        Args:
            input_size (int): Kích thước vector metrics
            hidden_size (int): Kích thước lớp ẩn
            learning_rate (float): Tốc độ học
        """
        self.meta_controller = MetaController(input_size, hidden_size)
        self.optimizer = torch.optim.Adam(self.meta_controller.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
    def update(self, metrics: List[float], target_performance: float) -> Tuple[float, float, float]:
        """
        Cập nhật meta-controller
        
        Args:
            metrics (List[float]): Vector metrics đầu vào
            target_performance (float): Hiệu suất mục tiêu
            
        Returns:
            Tuple[float, float, float]: Các hệ số điều chỉnh mới
        """
        # Chuyển đổi metrics thành tensor
        x = torch.tensor(metrics, dtype=torch.float32)
        target = torch.tensor(target_performance, dtype=torch.float32)
        
        # Forward pass
        factors = self.meta_controller(x)
        
        # Tính loss (giả sử factors càng gần 1.0 càng tốt)
        loss = self.criterion(factors, torch.ones_like(factors))
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return tuple(factors.detach().numpy())
        
    def save(self, path: str):
        """
        Lưu meta-controller
        
        Args:
            path (str): Đường dẫn file
        """
        torch.save({
            'meta_controller_state_dict': self.meta_controller.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
        
    def load(self, path: str):
        """
        Load meta-controller
        
        Args:
            path (str): Đường dẫn file
        """
        checkpoint = torch.load(path)
        self.meta_controller.load_state_dict(checkpoint['meta_controller_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 