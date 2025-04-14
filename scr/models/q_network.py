import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        """
        QNetwork được sử dụng để xấp xỉ Q-values cho mỗi trạng thái và hành động.
        
        Args:
            input_size (int): Kích thước vector trạng thái (số lượng đặc trưng của trạng thái).
            output_size (int): Số lượng hành động khả dụng.
        """
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, output_size)
    
    def forward(self, x):
        """
        Truyền xuôi của mạng, nhận đầu vào là trạng thái và trả về Q-values cho mỗi hành động.
        
        Args:
            x (torch.Tensor): Vector trạng thái với shape (batch_size, input_size).
        
        Returns:
            torch.Tensor: Q-values cho mỗi hành động với shape (batch_size, output_size).
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def save(self, path):
        """Lưu model."""
        torch.save(self.state_dict(), path)
    
    def load(self, path):
        """Load model."""
        self.load_state_dict(torch.load(path))
