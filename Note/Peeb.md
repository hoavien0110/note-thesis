# PEEB: Part-based Image Classifiers with an Explainable and Editable Language Bottleneck

## 1. Thành phần chính trong PEEB
## 1.1. Image Encoder (Bộ mã hóa ảnh)
- Chức năng: Trích xuất đặc trưng từ ảnh.
- Hoạt động:
    - Ảnh đầu vào được chia thành các patch nhỏ (ví dụ: 16×16).
    - Mỗi patch được mã hóa thành một embedding (vector đặc trưng) bởi OWL-ViT (một Vision Transformer).
    - Kết quả: Tập hợp các embedding đại diện cho các vùng (patches) trong ảnh.

## 1.2. Text Encoder (Bộ mã hóa văn bản)
- Chức năng: Biến các mô tả văn bản thành các embedding để so sánh với ảnh.
- Hoạt động:
    - Tên của các bộ phận (ví dụ: "back", "beak") và các mô tả chi tiết (ví dụ: "vibrant blue feathers") được mã hóa thành các text embeddings.
    - Kết quả: Một vector embedding cho mỗi mô tả.

## 1.3. Linear Projection
- Chức năng: Liên kết không gian embedding của ảnh và văn bản.
- Hoạt động:
    - Các embedding của ảnh từ Image Encoder được chuyển qua một lớp Linear Projection để ánh xạ sang cùng một không gian với embedding văn bản.
    - Điều này giúp mô hình so sánh trực tiếp giữa embedding ảnh và văn bản.

## 1.4. Part MLP (Multi-Layer Perceptron cho từng phần)
- Chức năng: So khớp embedding của ảnh với mô tả văn bản.
- Hoạt động:
    - Mỗi embedding ảnh tương ứng với một vùng (patch) được kiểm tra xem có giống với các mô tả bộ phận (e.g., "back: blue feathers") hay không.
    - Tính toán độ tương đồng giữa ảnh và mô tả.

## 1.5. Box MLP
- Chức năng: Dự đoán vị trí của các bộ phận trong ảnh.
- Hoạt động:
    - Dựa trên các embedding ảnh đã chọn, Box MLP dự đoán bounding box (tọa độ khung giới hạn) của từng bộ phận.

# 2. Quy trình hoạt động
## 2.1. Trích xuất embedding từ ảnh và văn bản
- Ảnh đầu vào được mã hóa thành các patch embeddings.
- Văn bản mô tả từng bộ phận được mã hóa thành text embeddings.

## 2.2. Tính toán độ tương đồng
- So sánh patch embeddings từ ảnh với text embeddings để xác định mức độ tương đồng.
- Với mỗi bộ phận (e.g., "back"), chọn patch có độ tương đồng cao nhất để đại diện cho bộ phận đó.

## 2.3. Phân loại
Dựa trên embedding đã chọn, tính tổng điểm tương đồng giữa tất cả các bộ phận và các mô tả văn bản cho từng lớp.
Lớp có tổng điểm cao nhất được chọn làm kết quả phân loại.
