import random

class ExperienceReplay:
    def __init__(self, capacity: int):
        """
        Khởi tạo bộ nhớ Experience Replay với dung lượng tối đa.
        
        Args:
            capacity (int): Số lượng transition tối đa có thể lưu trữ.
        """
        self.capacity = capacity
        self.memory = []

    def store_transition(self, transition: tuple):
        """
        Lưu trữ một transition vào bộ nhớ.
        
        Args:
            transition (tuple): Một tuple gồm (state, action, reward, next_state, done).
        """
        if len(self.memory) >= self.capacity:
            # Nếu bộ nhớ đầy, loại bỏ transition cũ nhất
            self.memory.pop(0)
        self.memory.append(transition)

    def sample(self, batch_size: int) -> list:
        """
        Lấy ngẫu nhiên một batch các transition từ bộ nhớ.
        
        Args:
            batch_size (int): Kích thước mẫu muốn lấy.
        
        Returns:
            list: Danh sách các transition được chọn.
        """
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self) -> int:
        """
        Trả về số lượng transition hiện có trong bộ nhớ.
        
        Returns:
            int: Số lượng transition.
        """
        return len(self.memory)
