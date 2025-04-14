class EarlyStopping:
    def __init__(self, patience=50, min_epsilon=0.01, min_improvement=0.01):
        """
        Khởi tạo EarlyStopping.
        
        Args:
            patience (int): Số episode không cải thiện trước khi dừng
            min_epsilon (float): Giá trị epsilon tối thiểu
            min_improvement (float): Cải thiện tối thiểu để coi là có tiến bộ
        """
        self.patience = patience
        self.min_epsilon = min_epsilon
        self.min_improvement = min_improvement
        self.best_win_rate = 0
        self.no_improvement_count = 0
        self.should_stop = False
        
    def check(self, win_rate, epsilon):
        """
        Kiểm tra điều kiện dừng.
        
        Args:
            win_rate (float): Tỷ lệ thắng hiện tại
            epsilon (float): Giá trị epsilon hiện tại
            
        Returns:
            bool: True nếu nên dừng, False nếu nên tiếp tục
        """
        # Kiểm tra epsilon
        if epsilon <= self.min_epsilon:
            print(f"Điều kiện dừng: Epsilon ({epsilon:.4f}) đã đạt mức tối thiểu ({self.min_epsilon})")
            self.should_stop = True
            return True
            
        # Kiểm tra tiến bộ
        if win_rate > self.best_win_rate + self.min_improvement:
            self.best_win_rate = win_rate
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1
            
        # Kiểm tra số lần không cải thiện
        if self.no_improvement_count >= self.patience:
            print(f"Điều kiện dừng: Không có tiến bộ sau {self.patience} lần đánh giá")
            self.should_stop = True
            return True
            
        return False
        
    def reset(self):
        """Reset trạng thái."""
        self.best_win_rate = 0
        self.no_improvement_count = 0
        self.should_stop = False 