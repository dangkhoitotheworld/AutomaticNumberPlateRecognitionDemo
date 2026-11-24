# 🚗 Automatic Number Plate Recognition Demo (YOLOv8 + OpenCV + SORT)

Ứng dụng demo **nhận diện biển số xe** từ ảnh và video, được thực hiện trong khuôn khổ tiểu luận nhóm môn *Chuyên đề Hệ thống giao thông thông minh* (đề tài 20: *Xây dựng ứng dụng nhận diện biển số xe*).

Hệ thống gồm hai mô hình YOLOv8 được **fine-tune** từ mô hình pre-trained: một mô hình **phát hiện biển số** (`plate.pt`) và một mô hình **nhận diện ký tự** (`char.pt`). Khi mô hình ký tự không đọc được, chương trình có thể dùng **RapidOCR** (pre-trained) làm phương án dự phòng.

---

## 📑 Mục lục

1. [Tính năng](#-tính-năng)
2. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
3. [Quy trình xử lý](#-quy-trình-xử-lý)
4. [Cài đặt](#-cài-đặt)
5. [Cách sử dụng](#-cách-sử-dụng)
6. [Huấn luyện mô hình](#-huấn-luyện-mô-hình)
7. [Dữ liệu và giấy phép](#-dữ-liệu-và-giấy-phép)
8. [Kết quả](#-kết-quả)
9. [Hạn chế và hướng phát triển](#-hạn-chế-và-hướng-phát-triển)
10. [Nhóm thực hiện](#-nhóm-thực-hiện)

---

## ✨ Tính năng

- Phát hiện vị trí biển số xe trong **ảnh** và **video**.
- Đọc ký tự trên biển số bằng mô hình YOLOv8 tự fine-tune (`char.pt`).
- Theo dõi (tracking) biển số qua các khung hình bằng thuật toán **SORT**, giúp hạn chế đọc lặp lại cùng một xe.
- Dự phòng bằng **RapidOCR** (pre-trained) khi cần.
- Ba chế độ chạy: `yolo`, `ocr`, `auto`.
- Có sẵn notebook Google Colab để huấn luyện lại mô hình.

## 📁 Cấu trúc thư mục

```
AutomaticNumberPlateRecognitionDemo/
├── bien_so_yolo.py                 # Chương trình chính: phát hiện + đọc biển số
├── plate.pt                        # Mô hình YOLOv8 phát hiện biển số
├── char.pt                         # Mô hình YOLOv8 nhận diện ký tự
├── huan_luyen_yolov8_colab.ipynb   # Notebook huấn luyện (Google Colab)
├── anh_thu_nhieu_bien.png          # Ảnh thử nghiệm (5 biển số, có biển bị xoay)
├── ket_qua/                        # Ảnh/video kết quả đầu ra
└── README.md
```

## 🔄 Quy trình xử lý

```
Ảnh / Video
    │
    ▼
[1] Phát hiện biển số  (YOLOv8 – plate.pt)
    │
    ▼
[2] Theo dõi biển số qua các khung hình  (SORT)
    │
    ▼
[3] Cắt vùng biển số  (OpenCV)
    │
    ▼
[4] Nhận diện ký tự  (YOLOv8 – char.pt)   ──(không đọc được)──▶  RapidOCR (dự phòng)
    │
    ▼
[5] Ghép ký tự theo vị trí → chuỗi biển số → vẽ kết quả lên ảnh/video
```

## ⚙️ Cài đặt

**Yêu cầu:** Python 3.9 trở lên (khuyến nghị 3.10+).

```bash
# 1. Tải mã nguồn
git clone https://github.com/phamdangkhoi170706-blip/AutomaticNumberPlateRecognitionDemo.git
cd AutomaticNumberPlateRecognitionDemo

# 2. (Tuỳ chọn) Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Cài thư viện
pip install -r requirements.txt
```

> Nếu chưa có `requirements.txt`, tham khảo mục thư viện chính: `ultralytics`, `opencv-python`, `numpy`, `rapidocr-onnxruntime`, cùng các thư viện mà bản cài đặt SORT yêu cầu (thường là `filterpy`, `scipy`).

## ▶️ Cách sử dụng

Chương trình chính là `bien_so_yolo.py`, hỗ trợ ba chế độ qua tham số `--mode`:

| Chế độ | Mô tả |
|--------|-------|
| `yolo` | Chỉ dùng hai mô hình tự fine-tune (`plate.pt` + `char.pt`). Dùng khi muốn đánh giá riêng chất lượng mô hình của nhóm. |
| `ocr`  | Phát hiện biển số bằng YOLO, đọc ký tự bằng RapidOCR (pre-trained). |
| `auto` | Ưu tiên mô hình YOLO ký tự; nếu không đọc được thì chuyển sang RapidOCR. |

Ví dụ (kiểm tra lại tên tham số bằng `python bien_so_yolo.py --help`):

```bash
# Chạy thử với ảnh mẫu
python bien_so_yolo.py --mode auto --source anh_thu_nhieu_bien.png

# Chỉ dùng mô hình tự fine-tune
python bien_so_yolo.py --mode yolo --source anh_thu_nhieu_bien.png

# Chạy trên video
python bien_so_yolo.py --mode auto --source duong_dan/video.mp4
```

Kết quả được lưu trong thư mục `ket_qua/`.

## 🧠 Huấn luyện mô hình

Hai mô hình được **fine-tune (huấn luyện tiếp) từ mô hình YOLOv8 pre-trained**, không huấn luyện từ đầu. Toàn bộ quá trình nằm trong notebook `huan_luyen_yolov8_colab.ipynb`:

1. Mở notebook trên Google Colab và bật GPU (`Runtime → Change runtime type`).
2. Tải bộ dữ liệu từ Roboflow (xem mục dưới).
3. Chạy các ô lệnh để huấn luyện mô hình phát hiện biển số và mô hình nhận diện ký tự.
4. Tải hai file trọng số `plate.pt` và `char.pt` về, đặt cùng thư mục với `bien_so_yolo.py`.

| Mô hình | Nhiệm vụ | Khởi tạo từ |
|---------|----------|-------------|
| `plate.pt` | Phát hiện biển số | YOLOv8 pre-trained |
| `char.pt`  | Nhận diện ký tự   | YOLOv8 pre-trained |

## 📊 Dữ liệu và giấy phép

| Bộ dữ liệu | Dùng cho | Nguồn | Giấy phép |
|------------|----------|-------|-----------|
| Dữ liệu biển số | Huấn luyện `plate.pt` | Roboflow – tác giả QuinnNg | CC BY 4.0 |
| Dữ liệu ký tự   | Huấn luyện `char.pt`  | Roboflow – `[điền tên/tác giả]` | `[điền giấy phép]` |

Các mô hình và thư viện bên thứ ba được sử dụng:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (giấy phép AGPL-3.0)
- [OpenCV](https://opencv.org/)
- [SORT](https://github.com/abewley/sort) – Simple Online and Realtime Tracking
- [RapidOCR](https://github.com/RapidAI/RapidOCR)

## 🏁 Kết quả

Ảnh minh hoạ nằm trong thư mục [`ket_qua/`](ket_qua/).

| Chỉ số | Mô hình phát hiện biển số | Mô hình ký tự |
|--------|---------------------------|----------------|
| Precision | `[ ]` | `[ ]` |
| Recall | `[ ]` | `[ ]` |
| mAP@0.5 | `[ ]` | `[ ]` |

Kết quả thử nghiệm thực tế (chế độ `--mode yolo`): `[điền số ảnh/video đã thử, tỉ lệ đọc đúng]`.

## ⚠️ Hạn chế và hướng phát triển

**Hạn chế**

- Độ chính xác giảm với biển số bị mờ, nghiêng nhiều, thiếu sáng hoặc bị che khuất.
- Dữ liệu huấn luyện có hạn nên chưa bao quát mọi loại biển (biển 2 dòng, biển vuông, nhiều kiểu chữ).
- Đây là bản demo phục vụ học tập, chưa tối ưu cho triển khai thời gian thực quy mô lớn.

**Hướng phát triển**

- Mở rộng và cân bằng dữ liệu huấn luyện, thêm các điều kiện ánh sáng và thời tiết khác nhau.
- Bổ sung bước kiểm tra định dạng biển số Việt Nam để hậu xử lý kết quả.
- Xây dựng giao diện web/ứng dụng và lưu kết quả vào cơ sở dữ liệu.
- Tích hợp với các bài toán giao thông thông minh như kiểm soát ra vào, thu phí tự động.
