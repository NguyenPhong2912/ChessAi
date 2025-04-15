# Chess AI with Advanced Deep Learning

Dự án này là một AI cờ vua thông minh sử dụng kết hợp nhiều kỹ thuật học sâu tiên tiến:
- MCTS (Monte Carlo Tree Search)
- Q-Learning với Deep Neural Network
- ADMCDL (Adaptive Meta-Controller Deep Learning)

## 🎮 Tính năng chính

1. **Giao diện trực quan**
   - Bàn cờ đẹp mắt với hiệu ứng animation
   - Hiển thị gợi ý nước đi hợp lệ
   - Panel thông tin hiển thị lượt đi và trạng thái game
   - Hỗ trợ kéo thả quân cờ với hiệu ứng mượt mà

2. **AI thông minh**
   - Sử dụng MCTS khi epsilon < 0.1 (200 lần mô phỏng)
   - Q-Learning với replay memory 50,000 và batch size 128
   - DNN với kiến trúc 1024 -> 512 -> 256 -> output_size
   - ADMCDL với 4 options cho các chiến lược khác nhau

3. **Luật chơi đầy đủ**
   - Hỗ trợ tất cả luật cờ vua chuẩn
   - Phong tốt với menu lựa chọn quân cờ
   - Chiếu tướng và các điều kiện hòa
   - Luật đặc biệt: khi chỉ còn vua, người ăn quân cuối thắng

## 📁 Cấu trúc dự án và chi tiết module

### 1. Game Logic (`scr/game/`)
- **ChessEngine.py** (11KB)
  - Xử lý logic cờ vua chính
  - Đánh giá vị trí và tính toán nước đi
  - Tích hợp với các agent AI
- **ChessRules.py** (17KB)
  - Kiểm tra tính hợp lệ của nước đi
  - Xử lý các trường hợp đặc biệt (phong tốt, chiếu)
  - Quản lý luật chơi
- **ChessBoard.py** (6.7KB)
  - Quản lý trạng thái bàn cờ
  - Xử lý di chuyển quân cờ
  - Lưu trữ lịch sử nước đi

### 2. AI Agents (`scr/agents/`)
- **q_learning_agent.py** (6.8KB)
  - Agent học Q-Learning chính
  - Tích hợp với replay memory
  - Xử lý epsilon-greedy exploration
- **mcts.py** (2.3KB)
  - Monte Carlo Tree Search
  - 200 simulations mỗi nước đi
  - UCB1 selection policy
- **experience_replay.py** (1.4KB)
  - Bộ nhớ replay với capacity 50,000
  - Batch sampling cho training
  - Prioritized experience replay

### 3. Neural Networks (`scr/models/`)
- **advanced_q_network.py** (12KB)
  - Mạng neural với ADMCDL
  - 4 options cho các chiến lược
  - Batch normalization và dropout
- **q_network.py** (1.3KB)
  - Mạng Q cơ bản
  - Kiến trúc 1024 -> 512 -> 256
- **ModelTrainer.py** (4.7KB)
  - Huấn luyện model với early stopping
  - Cập nhật target network
  - Lưu và load model

### 4. Training (`scr/training/`)
- **trainer.py** (4.4KB)
  - Huấn luyện AI với reward shaping
  - Xử lý episode và batch training
  - Theo dõi metrics
- **model_evaluator.py** (3.6KB)
  - Đánh giá model qua win rate
  - So sánh với các phiên bản
  - Logging kết quả
- **early_stopping.py** (2.0KB)
  - Dừng sớm khi không cải thiện
  - Theo dõi validation loss
  - Lưu model tốt nhất

### 5. ADMCDL (`scr/Adaptive Meta-Controller Deep Q-learning/`)
- **train.py** (8.6KB)
  - Huấn luyện ADMCDL
  - Meta-learning controller
  - Option training
- **meta_controller.py** (4.1KB)
  - Điều khiển meta-learning
  - Option selection
  - Policy adaptation
- **dqn_agent.py** (7.8KB)
  - Deep Q-Network agent
  - Experience replay
  - Target network
- **chess_env.py** (3.4KB)
  - Môi trường cờ vua
  - State representation
  - Reward calculation

### 6. Hỗ trợ
- **utils/**: Công cụ hỗ trợ
- **visualization/**: Hiển thị và animation
- **data/**: Lưu trữ dữ liệu huấn luyện
- **train_model.py**: Script huấn luyện chính

## 🔧 Yêu cầu hệ thống và thư viện

### Python 3.9+
- **PyTorch**: Deep learning framework
- **Pygame**: Giao diện đồ họa
- **python-chess**: Xử lý logic cờ vua
- **numpy**: Xử lý mảng và tính toán
- **pandas**: Xử lý dữ liệu và logging
- **matplotlib**: Vẽ đồ thị và visualization

### Cài đặt
```bash
# Clone repository
git clone [url]

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

## 🎯 Cách chơi

### 1. Chơi với AI
```bash
python main.py
```

**Điều khiển:**
- Click và kéo quân cờ để di chuyển
- Các nước đi hợp lệ sẽ được highlight màu xanh
- Thả quân cờ vào ô hợp lệ để thực hiện nước đi
- Khi tốt đến cuối bàn cờ, menu phong sẽ hiện ra
- Chọn quân cờ muốn phong (hậu, xe, tượng, mã)

**Luật chơi:**
- Chiếu hết: người chiếu thắng
- Hòa do bế tắc hoặc không còn nước đi
- Khi chỉ còn vua: người ăn quân cuối cùng thắng

### 2. Huấn luyện AI

#### Huấn luyện với UI
```bash
python notebook/train_chess_ai_with_ui.py
```
- Hiển thị bàn cờ và thông tin huấn luyện
- Theo dõi reward, win rate, epsilon
- Lưu model tốt nhất tự động

#### Huấn luyện không có UI
```bash
python scr/train_model.py
```
- Huấn luyện nhanh hơn
- Lưu log chi tiết
- Đánh giá model sau mỗi episode

**Tham số huấn luyện:**
- Số episode: 500
- Số nước đi tối đa: 200
- Thời gian tối đa: 600s
- Early stopping: 50 episode không cải thiện

## 📊 Đánh giá hiệu suất

1. **Win Rate**
   - Đánh giá qua 100 ván đấu
   - So sánh với các phiên bản trước

2. **Reward**
   - Tổng reward mỗi episode
   - Reward trung bình
   - Độ ổn định

3. **Thời gian**
   - Thời gian huấn luyện
   - Thời gian suy luận mỗi nước đi

## 📝 Liên hệ (contacts)

Nếu có góp ý hoặc cần hỗ trợ, vui lòng liên hệ:
- Personal Email: ntphong1231@gmail.com
- Working Email: Phong.2474802010304@vanlanguni.vn
## 📜 License

VLUVN - VANLANG UNIVERSITY VIETNAM - FACUTLY OF INFORMATION TECHNOLOGY VLU.  
AUTHOR : 
NGUYEN THANH PHONG, 
ASSOCIATES: 
- TIEN TRI NGUYEN
- NGUYEN HA THU
INSTRUCTOR:
NGUYEN THE AN,
