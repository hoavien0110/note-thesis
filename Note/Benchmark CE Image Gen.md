# Benchmarking Counterfactual Image Generation

# 1. Introduction
- Pearlian Framework:
    - Sử dụng mô hình nhân quả cấu trúc (Structural Causal Models - SCMs) để xác định các đường dẫn nhân quả giữa các biến cấp cao.
    - Kết hợp mô hình Deep-SCM và Abduction-Action-Prediction để đánh giá các phương pháp tạo hình ảnh phản thực.

- Contribution:
    - Khung đánh giá toàn diện:
        - Đánh giá hiệu quả các mô hình dựa trên SCMs.
        - Áp dụng trên các bộ dữ liệu khác nhau (tổng hợp, tự nhiên, y tế).
    - Mở rộng mô hình:
        - Kiểm tra trên các đồ thị nhân quả (causal graphs) và bộ dữ liệu mới.
        - Áp dụng HVAE và GAN trên đồ thị nhân quả phức tạp và hình ảnh y tế.
    - Tiêu chuẩn đánh giá:
        - Sử dụng các chỉ số để đo lường hiệu quả của hình ảnh phản thực, như:
            - Composition (tính cấu thành).
            - Effectiveness (hiệu quả).
            - Minimality (sự tối thiểu).
            - Realism (tính chân thực).
    - Công cụ hỗ trợ cộng đồng:
        - Cung cấp một gói Python dễ sử dụng để tích hợp các mô hình và phương pháp mới.

## 2. 