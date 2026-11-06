# Đề Cương Thực Nghiệm (Experiment Protocol)

## 1. Mốc thời gian cắt (Temporal Cutoff)
- **Cutoff Date**: $T = 31/12/2018$.
- Tuyệt đối không nạp y văn hay dữ liệu thử nghiệm sau năm 2018 vào quá trình huấn luyện và dựng đồ thị.

## 2. Chỉ số đo lường
- **Recall@K** ($K=5, 10$): Đo lường tỷ lệ các Hard Positives lọt vào Top K.
- **MRR (Mean Reciprocal Rank)**: Vị trí nghịch đảo của bệnh nhãn vàng đầu tiên.
- **NLP Precision**: Đánh giá trên mẫu ngẫu nhiên $100 - 200$ quan hệ trích xuất.
- **Ablation Study**:
  - Không có DirectionFit (coi mọi cạnh đều dương).
  - Không có Phạt mâu thuẫn (Contradiction Penalty).
  - Chỉ dùng Shortest Path thay vì Metapath DFS + Scoring.
  - Chỉ dùng NLP trích xuất không kèm đồ thị nền Hetionet.
