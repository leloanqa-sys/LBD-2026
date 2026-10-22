# Quy Chuẩn Lan Truyền Dấu & Đường Đi Vàng (Biological Rules)

## 1. Bảng quy đổi dấu quan hệ sinh học
| Loại quan hệ | Dấu ($s_i$) | Ý nghĩa sinh học | Ví dụ |
|---|---|---|---|
| INHIBITS | -1 | Ức chế, phong bế, kìm hãm | Dapagliflozin inhibits SLC5A2 |
| ACTIVATES | +1 | Kích hoạt, kích thích | TGF activates Vasoconstriction |
| CAUSES / INCREASES | +1 | Gây ra, tăng cường | Hyperfiltration causes CKD |
| DECREASES | -1 | Giảm thiểu, hạ bớt | Diuresis decreases plasma volume |
| ASSOCIATED_WITH | 0 | Tương quan, liên kết chưa rõ cơ chế | Gene associated with disease |
| TREATS / PALLIATES | -1 | Điều trị, giảm nhẹ triệu chứng | Drug treats disease |

## 2. Quy tắc nhân dấu toàn tuyến (Path Sign)
$$S_{path} = \prod_{i=1}^n s_i$$
- Nếu $S_{path} = -1$: Khớp chiều điều trị (Thuốc làm giảm tác nhân gây bệnh), hệ số $DirectionFit = 1.5$.
- Nếu chứa cạnh $0$: Đường đi gợi mở (Suggestive), hệ số $DirectionFit = 1.0$.
- Nếu $S_{path} = +1$: Ngược chiều điều trị (Nguy cơ gây độc/tác dụng phụ), hệ số $DirectionFit = 0.0$ và chuyển vào tập mâu thuẫn ($U_{neg}$).
