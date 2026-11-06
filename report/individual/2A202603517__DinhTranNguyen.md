# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603517**
- **Họ và tên: Đinh Trần Nguyên**
- **SEED** (in ở ô đầu notebook):

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | X |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | X |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | X |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            1.000   0.287      0.446        0.571   
2. BoW + MLP (deep learning)         0.988   0.908      0.946        0.947   
3a. Encoder (MiniLM) + kNN           0.903   0.966      0.933        0.947   
3b. Encoder (MiniLM) + LogReg        0.943   0.954      0.949        0.974   
3c. Encoder (MiniLM) + MLP head      0.955   0.966      0.960        0.974   
4a. LLM nhỏ zero-shot                0.731   0.950      0.826        0.826   
4b. LLM lớn zero-shot                0.826   0.950      0.884        0.884   
4c. LLM nhỏ few-shot                 0.792   0.950      0.864        0.864   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            9.924      13672.094       0.049   
2. BoW + MLP (deep learning)         7.502      34606.685       0.283   
3a. Encoder (MiniLM) + kNN         116.040         31.779      56.393   
3b. Encoder (MiniLM) + LogReg      116.622         52.447      29.018   
3c. Encoder (MiniLM) + MLP head    117.128         50.085      42.812   
4a. LLM nhỏ zero-shot                0.000          1.102     336.292   
4b. LLM lớn zero-shot                0.000          1.141     347.067   
4c. LLM nhỏ few-shot                 0.000          0.627    2258.362   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.253           0.0  
2. BoW + MLP (deep learning)          0.354           0.0  
3a. Encoder (MiniLM) + kNN           69.021           0.0  
3b. Encoder (MiniLM) + LogReg        37.136           0.0  
3c. Encoder (MiniLM) + MLP head      57.345           0.0  
4a. LLM nhỏ zero-shot              2410.553           0.0  
4b. LLM lớn zero-shot              2419.392           0.0  
4c. LLM nhỏ few-shot               2881.270           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.


## 3. Dự đoán và thực tế

| Dự đoán | Tôi đoán gì? | Kết quả thực tế | Vì sao lệch? |
|---|---|---|---|
| `ml_f1` | F1 khoảng 0.87, Precision cao hơn Recall. | F1 = 0.446, Precision = 1.0, Recall = 0.287. | Tôi đánh giá quá cao khả năng nhận diện Spam. Mô hình bỏ sót nhiều Spam khiến Recall thấp và kéo F1 giảm mạnh. |
| `dl_relu` | Bỏ ReLU sẽ khiến F1 giảm về gần Logistic Regression. | F1 không giảm mà đạt 0.952, cao hơn MLP có ReLU (0.946). | Tôi chưa xét đến ảnh hưởng của quá trình tối ưu và khởi tạo trọng số. Mô hình tuyến tính vẫn có thể hoạt động tốt trên dữ liệu BoW. |
| `ml_lr` | Với lr=50, loss sẽ dao động zigzag hoặc giảm đột ngột. | Loss tăng từ khoảng 0.7 lên 2.5 rồi giảm trở lại. | Learning rate quá lớn khiến bước cập nhật vượt qua vùng loss thấp. Tôi dự đoán đúng xu hướng mất ổn định nhưng chưa chính xác về diễn biến loss. |

## 4. Phân tích 50 case khó

Trong stress test 50 case khó, không có trường hợp nào mà tất cả mô hình đều dự đoán sai. Riêng LLM nhỏ sử dụng prompt phòng thủ (4d) sai 3/50 case, đạt Accuracy 94%.

Cả 3 lỗi đều thuộc nhóm `ham_looks_spam`, tức tin nhắn hợp lệ nhưng có đặc điểm giống Spam.

- **Case #2:** Tin nhắn thông báo pizza miễn phí tại văn phòng bị nhận nhầm là Spam do chứa các từ `FREE`, `right now`.
- **Case #9:** Tin nhắn tặng vé concert bị nhận nhầm là Spam do chứa `free tickets` và yêu cầu gửi địa chỉ.

Nguyên nhân có thể là mô hình quá nhạy với những dấu hiệu thường gặp trong Spam mà chưa phân biệt chính xác ngữ cảnh giao tiếp hợp lệ. Đây là lỗi False Positive, có nguy cơ chặn nhầm tin nhắn quan trọng của người dùng.

Kết quả cho thấy prompt phòng thủ không đảm bảo loại bỏ hoàn toàn lỗi phân loại, đặc biệt với những tin nhắn Ham có ngôn ngữ giống Spam.
## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

**Phương án lựa chọn:** Encoder (MiniLM) + Logistic Regression.

Theo phần phân tích trong notebook, mô hình có p95 khoảng 38 ms, dưới giới hạn 50 ms/tin, và F1 trên SUB khoảng 0.975. So với Embedding + kNN (khoảng 70 ms), phương án này có lợi thế về cả độ trễ và chất lượng.

Với 10 triệu SMS/ngày (trung bình khoảng 116 tin/giây), tôi đề xuất hệ thống kết hợp:

- **Tầng 1:** MiniLM + Logistic Regression xử lý các tin nhắn thông thường.
- **Tầng 2:** Những tin nhắn có xác suất không chắc chắn được chuyển sang bộ kiểm tra bổ sung hoặc LLM.

**Đánh đổi:** MiniLM cần tài nguyên tính toán để tạo embedding, đồng thời vẫn có thể sai với những tin nhắn spam được ngụy trang. Sử dụng thêm LLM có thể tăng khả năng phân tích ngữ cảnh nhưng làm tăng độ trễ và chi phí API.

Tôi ưu tiên mô hình có độ trễ thấp, chất lượng tốt và chi phí vận hành có thể kiểm soát, thay vì sử dụng LLM cho toàn bộ tin nhắn.

## 7. AI log và điều đã học

| Phần | Bạn hỏi AI gì (prompt) | AI trả lời/viết gì | Bạn đã sửa / kiểm chứng gì |
|---|---|---|---|
| **Part 0 – Data Exploration** | Hỏi về Data Leakage, Duplicate Data và cách tính Accuracy baseline. | Giải thích khái niệm, công thức và ảnh hưởng đến evaluation. | Tự phân tích ảnh hưởng của dữ liệu trùng lặp và đối chiếu với giải thích. |
| **Part 1 – ML** | Hỏi cú pháp NumPy để vectorize, tính sigmoid, gradient descent và các metrics. | Giải thích logic, hướng dẫn các hàm NumPy và cung cấp skeleton code tham khảo. | Triển khai các hàm theo cấu trúc notebook; có các ô kiểm thử tương ứng. |
| **Part 2 – Deep Learning** | Hỏi cú pháp PyTorch để xây dựng MLP và thực hiện training loop. | Hướng dẫn sử dụng `nn.Sequential`, các lớp mạng, optimizer, loss và backward propagation. | Triển khai MLP và training loop; giải thích hiện tượng tích lũy gradient. |
| **Part 3 – Embedding** | Hỏi cú pháp tính Cosine Similarity, tìm Top-K và triển khai kNN. | Hướng dẫn sử dụng NumPy, phép tính vector, sắp xếp độ tương đồng và voting. | Triển khai các hàm tương ứng trong notebook và có các ô kiểm thử. |
| **Part 4 – LLM** | Hỏi cách sử dụng cú pháp API, xây dựng prompt và xử lý phản hồi của LLM. | Hỗ trợ cấu trúc lời gọi API, tham số và logic phân loại zero-shot/few-shot. | Triển khai phần gọi API và prompt classification trong notebook. |

**Điều tôi học được:**

1. Trước đây tôi thường đánh giá mô hình dựa trên Accuracy, nhưng giờ hiểu rằng dữ liệu mất cân bằng và Data Leakage có thể khiến kết quả đánh giá sai lệch. Precision, Recall và F1 giúp phản ánh chất lượng phân loại Spam tốt hơn.
2. Tôi hiểu rõ hơn sự khác biệt giữa ML, DL, Embedding và LLM, đặc biệt là cách biểu diễn văn bản và khả năng nhận diện ngữ nghĩa của từng phương pháp.
3. Tôi nhận ra mô hình có F1 cao trên tập test thông thường vẫn có thể mắc lỗi trên các trường hợp khó. Khi triển khai thực tế cần cân nhắc cả độ trễ, chi phí và rủi ro chặn nhầm tin nhắn hợp lệ.
