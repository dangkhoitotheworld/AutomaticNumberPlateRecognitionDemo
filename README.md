# 🚗 Automatic Number Plate Recognition Demo

> Nhận diện biển số xe từ **ảnh** và **video** bằng YOLOv8, OpenCV, SORT và RapidOCR.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)
![License](https://img.shields.io/badge/License-MIT-green)

Đây là ứng dụng demo thuộc tiểu luận nhóm môn **Chuyên đề Hệ thống giao thông thông minh** (*Xây dựng ứng dụng nhận diện biển số xe*).

Hệ thống dùng hai mô hình YOLOv8 được **fine-tune** từ trọng số pre-trained:

| Mô hình | Nhiệm vụ |
| --- | --- |
| `plate.pt` | Phát hiện vị trí biển số trong ảnh/khung hình |
| `char.pt` | Nhận diện từng ký tự trên biển số |

Khi `char.pt` không đọc được, chương trình có thể chuyển sang **RapidOCR** (pre-trained) làm phương án dự phòng.

---

## 📑 Mục lục

1. [Tính năng](#-tính-năng)
2. [Quy trình xử lý](#-quy-trình-xử-lý)
3. [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
4. [Cài đặt](#️-cài-đặt)
5. [Cách sử dụng](#️-cách-sử-dụng)
6. [Huấn luyện mô hình](#-huấn-luyện-mô-hình)
7. [Dữ liệu](#-dữ-liệu)
8. [Kết quả](#-kết-quả)
9. [Hạn chế và hướng phát triển](#️-hạn-chế-và-hướng-phát-triển)

---

## ✨ Tính năng

- Phát hiện biển số trong **ảnh** và **video**.
- Đọc ký tự bằng mô hình YOLOv8 tự fine-tune (`char.pt`).
- Theo dõi biển số qua các khung hình bằng **SORT**, tránh đọc lặp lại cùng một xe.
- Dự phòng bằng **RapidOCR** khi mô hình ký tự không đọc được.
- Ba chế độ chạy: `yolo`, `ocr`, `auto`.
- Có sẵn notebook Google Colab để huấn luyện lại.

## 🔄 Quy trình xử lý

```
Ảnh / Video
    │
    ▼
[1] Phát hiện biển số           (YOLOv8 – plate.pt)
    │
    ▼
[2] Theo dõi qua các khung hình (SORT)
    │
    ▼
[3] Cắt vùng biển số            (OpenCV)
    │
    ▼
[4] Nhận diện ký tự             (YOLOv8 – char.pt) ──(thất bại)──▶ RapidOCR
    │
    ▼
[5] Ghép ký tự theo vị trí → chuỗi biển số → vẽ kết quả lên ảnh/video
```

## 📁 Cấu trúc thư mục

```
AutomaticNumberPlateRecognitionDemo/
├── bien_so_yolo.py                 # Chương trình chính
├── plate.pt                        # Trọng số phát hiện biển số
├── char.pt                         # Trọng số nhận diện ký tự
├── huan_luyen_yolov8_colab.ipynb   # Notebook huấn luyện (Colab)
├── anh_thu_nhieu_bien.png          # Ảnh thử nghiệm (5 biển, có biển bị xoay)
├── ket_qua/                        # Ảnh/video kết quả đầu ra
├── requirements.txt                # Thư viện cần cài
├── LICENSE
└── README.md
```

## ⚙️ Cài đặt

**Yêu cầu:** Python 3.9 trở lên (khuyến nghị 3.10+). Có GPU NVIDIA sẽ chạy video nhanh hơn nhưng không bắt buộc.

```bash
# 1. Tải mã nguồn
git clone https://github.com/dangkhoitotheworld/AutomaticNumberPlateRecognitionDemo.git
cd AutomaticNumberPlateRecognitionDemo

# 2. (Khuyến nghị) Tạo môi trường ảo
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Cài thư viện
pip install -r requirements.txt
```

Thư viện sử dụng: `ultralytics`, `opencv-python`, `numpy`, `rapidocr-onnxruntime`, `filterpy`, `scipy`.

## ▶️ Cách sử dụng

Chương trình chính là `bien_so_yolo.py`, chọn chế độ qua tham số `--mode`:

| Chế độ | Mô tả |
| --- | --- |
| `yolo` | Chỉ dùng hai mô hình tự fine-tune (`plate.pt` + `char.pt`). Phù hợp để đánh giá riêng chất lượng mô hình của nhóm. |
| `ocr` | Phát hiện biển bằng YOLO, đọc ký tự bằng RapidOCR. |
| `auto` | Ưu tiên `char.pt`; nếu không đọc được thì chuyển sang RapidOCR. |

```bash
# Ảnh mẫu, chế độ auto
python bien_so_yolo.py --mode auto --source anh_thu_nhieu_bien.png

# Chỉ dùng mô hình tự fine-tune
python bien_so_yolo.py --mode yolo --source anh_thu_nhieu_bien.png

# Video
python bien_so_yolo.py --mode auto --source duong_dan/video.mp4
```

Xem toàn bộ tham số: `python bien_so_yolo.py --help`. Kết quả được lưu trong thư mục `ket_qua/`.

## 🧠 Huấn luyện mô hình

Hai mô hình được **fine-tune từ YOLOv8 pre-trained**, không huấn luyện từ đầu. Toàn bộ quy trình nằm trong `huan_luyen_yolov8_colab.ipynb`:

1. Mở notebook trên Google Colab và bật GPU (`Runtime → Change runtime type`).
2. Tải bộ dữ liệu từ Roboflow (xem mục dưới).
3. Chạy các ô lệnh để huấn luyện mô hình phát hiện biển số và mô hình ký tự.
4. Tải `plate.pt` và `char.pt` về, đặt cùng thư mục với `bien_so_yolo.py`.

## 📊 Dữ liệu

| Bộ dữ liệu | Dùng cho | Nguồn | Giấy phép |
| --- | --- | --- | --- |
| Dữ liệu biển số | Huấn luyện `plate.pt` | Roboflow – tác giả QuinnNg | CC BY 4.0 |
| Dữ liệu ký tự | Huấn luyện `char.pt` | Roboflow | |

Thư viện và mô hình bên thứ ba:

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) – AGPL-3.0
- [OpenCV](https://opencv.org/)
- [SORT](https://github.com/abewley/sort) – Simple Online and Realtime Tracking
- [RapidOCR](https://github.com/RapidAI/RapidOCR)


## ⚠️ Hạn chế và hướng phát triển

**Hạn chế**

- Độ chính xác giảm với biển mờ, nghiêng nhiều, thiếu sáng hoặc bị che khuất.
- Dữ liệu huấn luyện còn hạn chế, chưa bao quát biển 2 dòng, biển vuông, nhiều kiểu chữ.
- Là bản demo học tập, chưa tối ưu cho triển khai thời gian thực quy mô lớn.

**Hướng phát triển**

- Mở rộng và cân bằng dữ liệu, thêm nhiều điều kiện ánh sáng và thời tiết.
- Hậu xử lý bằng kiểm tra định dạng biển số Việt Nam (regex).
- Xây dựng giao diện web/ứng dụng và lưu kết quả vào cơ sở dữ liệu.
- Tích hợp vào các bài toán như kiểm soát ra vào, thu phí tự động.
