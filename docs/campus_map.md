# Bản đồ Làng Đại học (ĐHQG-HCM): tên chuẩn và tọa độ

Tài liệu tham chiếu cho các thành phần được vẽ trong dashboard `web/station/index.html`.
Mục đích: có tên chính thức và tọa độ thật để đặt nhãn đúng, và để biết bố cục 3D hiện tại
lệch bao xa so với địa lý.

**Nguồn:** OpenStreetMap qua Overpass API, bbox `10.855,106.770,10.900,106.845`, truy vấn
2026-10-21; các tên riêng và tên đường được đối chiếu lại bằng Nominatim (tra xuôi và tra ngược). Tọa độ là tâm đối tượng OSM (node, hoặc
center của way/relation), hệ WGS84.

Không dùng Google Maps: không có API tra cứu hợp lệ cho việc này (Places API trả phí và điều
khoản hạn chế lưu tọa độ), còn OSM có tên tiếng Việt đầy đủ cho khu vực này.

## 1. Đối chiếu thành phần trong dashboard

Cột `scene (x, z)` là vị trí hiện tại trong `BUILDINGS`/`LABELS` của `index.html`.

| Nhãn trong dashboard | Tên chính thức | Lat | Lon | Đối tượng OSM | Tin cậy | scene (x, z) |
|---|---|---|---|---|---|---|
| Ga Metro Suối Tiên | Ga Đại học Quốc gia (tuyến metro số 1) | 10.86630 | 106.80120 | node `station` + relation | cao | (-98, 96) |
| ĐH Quốc tế | Trường Đại học Quốc tế (IU) | 10.87768 | 106.80128 | way `university` | cao | (-92, 26) |
| Nhà văn hóa Sinh viên | Nhà Văn hóa Sinh viên ĐHQG | 10.87514 | 106.80071 | way `community_centre` | cao | (-85, -25) |
| ĐH Khoa học Tự nhiên | Trường Đại học Khoa học tự nhiên (cơ sở 2) | 10.87620 | 106.79695 | way `university` | cao | (-40, -32) |
| Nhà điều hành ĐHQG | Nhà Điều hành ĐHQG | 10.86896 | 106.79623 | way `commercial` | trung bình, xem §2.4 | (-20, -25) |
| ĐH Bách Khoa | Trường Đại học Bách khoa (cơ sở 2) | 10.88022 | 106.80570 | way `university` | cao | (-40, 22) |
| ĐH KHXH&NV | Trường Đại học Khoa học Xã hội và Nhân văn (cơ sở 2) | 10.87345 | 106.80231 | node + way `university` | cao | (-18, 22) |
| Hồ Đá | Hồ đá 01 (cụm Hồ Đá 01/03/05/06/07) | 10.88317 | 106.79450 | relation `water` | trung bình, xem §2.3 | (-30, -80) |
| ĐH Kinh tế – Luật | Trường Đại học Kinh tế – Luật (UEL) | 10.87120 | 106.77844 | way `school` | cao | (22, -30) |
| TT Giáo dục Quốc phòng | Trung tâm Giáo dục Quốc phòng và An ninh | 10.88958 | 106.80200 | way `school` | cao | (26, 42) |
| ĐH Công nghệ Thông tin | Trường Đại học Công nghệ Thông tin (UIT) | 10.86999 | 106.80343 | way `university` | cao | (27, -80) |
| KTX khu A | Ký túc xá Khu A ĐHQG | 10.87778 | 106.80564 | node + way, kèm A1–A18, AH1–AH2, H1–H4 | cao | (78, -36) |
| Thư viện Trung tâm | Thư viện Trung tâm ĐHQG | 10.86978 | 106.79620 | way `library` | cao | (80, 32) |
| ĐH Khoa học Sức khỏe | Trường Đại học Khoa học Sức khỏe | 10.88864 | 106.79885 | node + way `university` | cao | (124, 16) |
| Khu Công nghệ Phần mềm | Khu Công nghệ Phần mềm ĐHQG (ITP) | 10.86829 | 106.79401 | way `research_institute` | cao | (124, -33) |
| Bến xe buýt ĐHQG (depot) | Bến xe buýt Đại học Quốc gia | 10.88418 | 106.77873 | node `bus_station` | cao | (-30, 42) |
| KTX khu B | Ký túc xá Khu B ĐHQG | 10.88374 | 106.78110 | way `dormitory`, kèm B4–B5, BA1–BA5, C1–C6, D2–D6, E1–E4 | cao | (-88, -82) |

## 2. Tên cần sửa

1. **Ga Metro Suối Tiên → Ga Đại học Quốc gia.** Ga metro phục vụ campus là
   *Ga Đại học Quốc gia* (10.86630, 106.80120). Ga Suối Tiên thật là *Bến xe Suối Tiên*
   (10.87953, 106.81406), cách campus khoảng 1,5 km về phía đông nam, cạnh Bến xe Miền Đông
   mới. Tên tuyến T1 trong dashboard ("Ga Suối Tiên ⇄ KTX khu B") nên đổi thành
   "Ga Đại học Quốc gia ⇄ KTX khu B".
2. **Bến xe buýt ĐHQG có hai bến.** Bến chính ở 10.88418, 106.77873 (góc tây bắc, cạnh KTX
   khu B) và *Bến xe khu A ĐHQG (mới)* ở 10.87449, 106.80763 (phía đông nam, cạnh KTX khu A).
   Dashboard đang gộp thành một depot; nếu chọn bến chính thì nó nằm ở góc tây bắc chứ không
   phải giữa bắc như scene hiện tại.
3. **Hồ Đá là một cụm năm hồ**, không phải một hồ: 01 (10.88317, 106.79450), 03 (10.88146,
   106.79884), 05 (10.87762, 106.79042), 06 (10.87822, 106.78480), 07 (10.88125, 106.78854).
   Hồ lớn nhất là hồ 01.
4. **Nhà điều hành ĐHQG:** OSM có hai ứng viên gần nhau, 10.86896, 106.79623 (`Nhà Điều Hành
   ĐHQG`, way `commercial`) và 10.87185, 106.79533 (`Nhà Điều Hành`, way). Cần xác nhận thực
   địa trước khi chốt.
5. **Tên trường nên ghi kèm cơ sở.** Bách khoa, Khoa học Tự nhiên, KHXH&NV trong Làng Đại học
   đều là cơ sở 2 (cơ sở chính ở trung tâm thành phố). Trên bản đồ nên ghi rõ để không lẫn.
6. **Cổng Bắc KTX khu A** (stop của tuyến T3) chưa có đối tượng trong OSM; phải xác định thực địa.

7. **Nhãn "Xa lộ Hà Nội" giữ nguyên.** OSM vẫn đặt tên đoạn chạy qua Làng Đại học và Bến xe
   Suối Tiên là *Xa lộ Hà Nội* (`highway=trunk`, tra ngược tại 10.8795, 106.8141). Tên *Võ
   Nguyên Giáp* trong OSM chỉ áp cho đoạn phía nam, khoảng 10.82–10.85 (Phường Thủ Đức). Nếu
   muốn ghi tên hành chính hiện hành thì ghi "Võ Nguyên Giáp (Xa lộ Hà Nội)".

## 3. Bố cục scene hiện tại là sơ đồ, không phải bản đồ

Đo 17 thành phần trên: tương quan giữa trục scene và trục địa lý gần như bằng 0
(`scene_x` với đông: +0,08; `scene_z` với bắc: -0,01), và 12/17 thành phần lệch hướng
hơn 60° so với vị trí thật. Ví dụ Thư viện Trung tâm nằm ở phía bắc trong scene nhưng thực tế
ở phía nam so với tâm campus; KTX khu B ở tây nam scene nhưng thực tế ở tây bắc.

Điều này khớp với chú thích trong mã dashboard (`campus layout (stylised, not to scale)`, `index.html`). Hệ quả: không
thể thay tọa độ thật vào bằng cách nhân tỉ lệ. Muốn theo địa lý phải vẽ lại toàn bộ `BUILDINGS`
và `ROUTE_DEFS` theo một phép chiếu đơn giản:

```
x = (lon - lon0) * 111320 * cos(lat0) * scale
z = -(lat - lat0) * 110574 * scale
```

với `lat0 = 10.87672`, `lon0 = 106.79631` (tâm campus theo 17 thành phần ở §1).

## 4. Thành phần thật chưa có trong dashboard

| Tên | Lat | Lon | Ghi chú |
|---|---|---|---|
| Trường Đại học Nông Lâm TP.HCM | 10.87236 | 106.78984 | trong Làng Đại học nhưng không thuộc ĐHQG |
| Trường Đại học Thể dục Thể thao TP.HCM | 10.87202 | 106.79584 | cùng khu |
| Trường Đại học An ninh Nhân dân | 10.87237 | 106.80522 | cùng khu |
| Bến xe Miền Đông mới | 10.87955 | 106.81619 | sát góc đông nam campus |
| Ga Bến xe Suối Tiên (ga cuối tuyến metro số 1) | 10.87953 | 106.81406 | bến xe, không phải ga ĐHQG |
| Ga Metro Khu Công nghệ cao | 10.85864 | 106.78807 | ga kế tiếp về phía tây nam |
| Khu đô thị Đại học Quốc gia | 10.87887 | 106.79402 | way, khu dân cư trong campus |

## 5. Tái lập truy vấn

```overpassql
[out:json][timeout:90];
(
  nwr["amenity"~"university|college|dormitory|library|bus_station"](10.855,106.770,10.900,106.845);
  nwr["building"="dormitory"](10.855,106.770,10.900,106.845);
  nwr["railway"~"station|halt|subway_entrance"](10.855,106.770,10.900,106.845);
  nwr["natural"="water"](10.855,106.770,10.900,106.845);
  nwr["name"~"Đại học|Ký túc xá|ĐHQG|Suối Tiên|Hồ Đá|Thư viện|Khu công nghệ|Giáo dục Quốc phòng|Nhà văn hóa|Nhà điều hành",i](10.855,106.770,10.900,106.845);
);
out center tags;
```

Gửi bằng POST tới `https://overpass-api.de/api/interpreter` với tham số `data=<query>`, kèm
`User-Agent`. Tra một địa điểm cụ thể: `https://nominatim.openstreetmap.org/search?q=<tên>&format=jsonv2&countrycodes=vn`.
