# Báo cáo cuối — standard-v1

Artifact được gom nguyên bản từ `runs/standard-v1/`, không chạy lại mô phỏng hay dựng lại báo cáo.

- [Báo cáo HTML](final_results.html)
- [Báo cáo Markdown](final_results.md)
- [Báo cáo PDF](final_results.pdf)
- [Biểu đồ và hình PNG](figures/): 9 hình xuất nguyên dữ liệu nhúng trong HTML.
- [Bảng số liệu CSV](tables/): summary/comparisons của study và masking, cùng 25 bảng trích từ HTML theo thứ tự xuất hiện.
- [Protocol và nguồn gốc](provenance/)
- [Giải thích bài toán](problem.html)
- [Bộ báo cáo trước khi gom](archive/before-standard-v1-collection/README.md)

`report.md` và `report.pdf` là liên kết tương thích cho thanh điều hướng của HTML.
Dữ liệu gốc từng ngày, metadata và lịch sử train giữ ở [runs/standard-v1/evidence](../runs/standard-v1/evidence/).
`collection.json` ghi nguồn và SHA-256 của các artifact đã gom; manifest trong provenance là bản gốc của bundle nguồn.

Lưu ý: PDF được sao chép nguyên bản có sẵn; các chỉnh sửa gần đây về bảng 1.1 và tách 3 biểu đồ nằm trong HTML/Markdown, chưa được dựng lại vào PDF.
Nội dung lịch sử trong `archive/` được giữ lại. Các đường dẫn lịch sử khác trong repository có thể vẫn trỏ tới bố cục reports cũ.
