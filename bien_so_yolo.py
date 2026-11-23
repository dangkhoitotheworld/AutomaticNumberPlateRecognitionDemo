import argparse
import csv
import math
import os
import re
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
VID_EXT = (".mp4", ".avi", ".mov", ".mkv")
OUT_DIR = "ket_qua"
MAX_SIDE = 2400          # ảnh lớn hơn sẽ được thu nhỏ khi chạy OCR để nhanh hơn
TILE = 1000              # kích thước ô khi chia nhỏ ảnh lớn (OCR)
ROTATIONS = (-25, 25, -45, 45, 90, -90, 180)  # góc xoay thử thêm khi OCR chưa thấy biển nào

# Biển số Việt Nam không dùng các chữ I, J, O, Q, W
DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"}
LETTER_FIX = {"0": "D", "8": "B", "5": "S", "2": "Z", "4": "A"}


# --------------------------------------------------------------------------- #
#  Đọc / ghi ảnh (hỗ trợ đường dẫn có dấu tiếng Việt)
# --------------------------------------------------------------------------- #
def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite(path, img):
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".jpg", img)
    if ok:
        buf.tofile(path)


# --------------------------------------------------------------------------- #
#  Chuẩn hóa và định dạng biển số
# --------------------------------------------------------------------------- #
def clean(txt):
    return re.sub(r"[^A-Z0-9]", "", txt.upper().replace("Đ", "D"))


def fix_chars(t):
    """Sửa các ký tự hay bị nhầm theo vị trí (2 số đầu = mã tỉnh, ký tự thứ 3 = chữ, đuôi = số)."""
    t = list(t)
    for i in range(min(2, len(t))):
        t[i] = DIGIT_FIX.get(t[i], t[i])
    if len(t) > 2:
        t[2] = LETTER_FIX.get(t[2], t[2])
    start = len(t) - (5 if len(t) >= 9 else 4)
    for i in range(max(3, start), len(t)):
        t[i] = DIGIT_FIX.get(t[i], t[i])
    return "".join(t)


HEAD_RE = re.compile(r"\d{2}[A-Z]{1,2}\d?")      # dòng trên của biển 2 dòng: 15P1, 29MD1, 59X1
ONE_RE = re.compile(r"(\d{2}[A-Z]{1,2})(\d{4,6})")  # biển 1 dòng: 51A12345, 29LD12345


def parse_head(c):
    if not 3 <= len(c) <= 5:
        return None
    t = list(c)
    for i in (0, 1):
        t[i] = DIGIT_FIX.get(t[i], t[i])
    t[2] = LETTER_FIX.get(t[2], t[2])
    t = "".join(t)
    return t if HEAD_RE.fullmatch(t) else None


def parse_tail(c):
    if not 4 <= len(c) <= 5:
        return None
    t = "".join(DIGIT_FIX.get(ch, ch) for ch in c)
    return t if t.isdigit() else None


def parse_single(c):
    if not 7 <= len(c) <= 10:
        return None
    m = ONE_RE.fullmatch(fix_chars(c))
    return (m.group(1), m.group(2)) if m else None


def fmt_num(d):
    return d[:3] + "." + d[3:] if len(d) == 5 else d


def fmt_two(head, tail):
    return f"{head[:2]}-{head[2:]} {fmt_num(tail)}"


def fmt_single(head, digits):
    if len(digits) == 6:  # số sê-ri dính liền (vd 59X1 + 12345)
        return fmt_two(head + digits[0], digits[1:])
    return f"{head}-{fmt_num(digits)}"


# --------------------------------------------------------------------------- #
#  Hình học
# --------------------------------------------------------------------------- #
def order_pts(p):
    """Sắp xếp 4 góc: trái-trên, phải-trên, phải-dưới, trái-dưới."""
    p = np.array(p, dtype=np.float32)
    s, d = p.sum(axis=1), p[:, 1] - p[:, 0]
    return np.array([p[np.argmin(s)], p[np.argmin(d)], p[np.argmax(s)], p[np.argmax(d)]], dtype=np.float32)


def warp_quad(img, quad):
    """Cắt vùng 4 góc rồi nắn thẳng thành hình chữ nhật nằm ngang."""
    tl, tr, br, bl = order_pts(quad)
    w = int(round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    h = int(round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    if w < 8 or h < 8:
        return None
    M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], np.float32),
                                    np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32))
    out = cv2.warpPerspective(img, M, (w, h))
    return cv2.rotate(out, cv2.ROTATE_90_COUNTERCLOCKWISE) if h > w else out


def expand(points, grow=1.12):
    """Khung xoay nhỏ nhất bao quanh các điểm, nới ra một chút."""
    (cx, cy), (w, h), a = cv2.minAreaRect(np.asarray(points, np.float32))
    return order_pts(cv2.boxPoints(((cx, cy), (w * grow + 4, h * grow * 1.08 + 4), a)))


def bbox(q):
    x, y, w, h = cv2.boundingRect(np.asarray(q, np.float32).astype(np.int32))
    return (x, y, x + w, y + h)


def iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua else 0.0


# --------------------------------------------------------------------------- #
#  Đọc chữ và ghép thành biển số
# --------------------------------------------------------------------------- #
class Line:
    """Một dòng chữ mà bộ OCR đọc được, kèm hình học để ghép dòng trên + dòng dưới."""

    def __init__(self, quad, text, conf):
        self.q = np.asarray(quad, np.float32)
        self.text, self.conf = text, conf
        self.clean = clean(text)
        self.c = self.q.mean(axis=0)
        v = self.q[1] - self.q[0]
        self.w = float(np.linalg.norm(v)) + 1e-6
        self.h = float(np.linalg.norm(self.q[3] - self.q[0])) + 1e-6
        self.u = v / self.w
        self.n = np.array([-self.u[1], self.u[0]], np.float32)
        self.ang = math.degrees(math.atan2(self.u[1], self.u[0]))


def read_lines(engine, img):
    res, _ = engine(img)
    return [Line(b, t, float(c)) for b, t, c in (res or []) if t and t.strip()]


def lines_to_plates(lines):
    """Ghép các dòng chữ thành biển số: (dòng trên + dòng dưới) hoặc 1 dòng."""
    heads = [(i, parse_head(l.clean)) for i, l in enumerate(lines)]
    tails = [(i, parse_tail(l.clean)) for i, l in enumerate(lines)]
    pairs = []
    for i, hd in heads:
        if not hd:
            continue
        a = lines[i]
        for j, tl in tails:
            if not tl or j == i:
                continue
            b = lines[j]
            d = b.c - a.c
            hav = (a.h + b.h) / 2.0
            across, along = float(d @ a.n), abs(float(d @ a.u))
            dang = abs((a.ang - b.ang + 180) % 360 - 180)
            if (0.5 * hav <= across <= 2.6 * hav and along <= 0.7 * max(a.w, b.w) and dang <= 20
                    and 0.3 <= a.w / b.w <= 2.2 and 0.5 <= a.h / b.h <= 2.0):
                pairs.append((abs(across - 1.4 * hav) + along, i, j, hd, tl))
    plates, used = [], set()
    for _, i, j, hd, tl in sorted(pairs, key=lambda t: t[0]):
        if i in used or j in used:
            continue
        used |= {i, j}
        a, b = lines[i], lines[j]
        plates.append(dict(quad=expand(np.vstack([a.q, b.q])), text=fmt_two(hd, tl),
                           conf=(a.conf + b.conf) / 2, kind="2 dòng"))
    for i, l in enumerate(lines):
        if i in used:
            continue
        p = parse_single(l.clean)
        if p:
            plates.append(dict(quad=expand(l.q), text=fmt_single(*p), conf=l.conf, kind="1 dòng"))
    return plates


def starts(length, size, step):
    if length <= size:
        return [0]
    s = list(range(0, length - size, step))
    return s + [length - size]


def pass_tiles(engine, img):
    """Ảnh lớn: chia thành các ô chồng lấn để thấy được biển số nhỏ."""
    H, W = img.shape[:2]
    out = []
    step = int(TILE * 0.75)
    for y in starts(H, TILE, step):
        for x in starts(W, TILE, step):
            for p in lines_to_plates(read_lines(engine, img[y:y + TILE, x:x + TILE])):
                p["quad"] = p["quad"] + np.array([x, y], np.float32)
                out.append(p)
    return out


def pass_candidates(engine, img):
    """Tìm các khung giống biển số bằng OpenCV, nắn thẳng rồi đọc riêng từng khung."""
    out = []
    for cand in plates_opencv(img):
        crop = cv2.copyMakeBorder(cand["crop"], 12, 12, 12, 12, cv2.BORDER_REPLICATE)
        found = lines_to_plates(read_lines(engine, crop))
        if found:
            best = max(found, key=lambda p: p["conf"])
            best["quad"] = cand["quad"]
            out.append(best)
    return out


def pass_rotate(engine, img, ang):
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1])
    nW, nH = int(H * s + W * c), int(H * c + W * s)
    M[0, 2] += nW / 2.0 - W / 2.0
    M[1, 2] += nH / 2.0 - H / 2.0
    rot = cv2.warpAffine(img, M, (nW, nH), borderValue=(114, 114, 114))
    Minv = cv2.invertAffineTransform(M)
    plates = lines_to_plates(read_lines(engine, rot))
    for p in plates:
        p["quad"] = order_pts(cv2.transform(p["quad"].reshape(-1, 1, 2), Minv).reshape(-1, 2))
    return plates


def dedupe(plates):
    keep = []
    for p in sorted(plates, key=lambda p: (not p.get("valid", True), -p["conf"])):
        bb = bbox(p["quad"])
        size = max(bb[2] - bb[0], bb[3] - bb[1])
        dup = False
        for k in keep:
            kb = bbox(k["quad"])
            dist = np.linalg.norm(p["quad"].mean(axis=0) - k["quad"].mean(axis=0))
            if iou(bb, kb) > 0.2 or (p["text"] == k["text"] and dist < size):
                dup = True
                break
        if not dup:
            keep.append(p)
    return sorted(keep, key=lambda p: (float(p["quad"][:, 0].mean())))  # từ trái sang phải


def analyze(engine, img):
    """Trả về list biển số: dict(text, conf, kind, quad) với quad ở toạ độ ảnh gốc."""
    H, W = img.shape[:2]
    scale = MAX_SIDE / float(max(H, W)) if max(H, W) > MAX_SIDE else 1.0
    work = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img

    plates = lines_to_plates(read_lines(engine, work))
    if max(work.shape[:2]) > 1300:
        plates += pass_tiles(engine, work)
    plates += pass_candidates(engine, work)
    if not plates:
        for ang in ROTATIONS:
            plates += pass_rotate(engine, work, ang)
            if plates:
                break
    plates = dedupe(plates)
    for p in plates:
        p["quad"] = (p["quad"] / scale).astype(np.float32)
    return plates


# --------------------------------------------------------------------------- #
#  Tìm khung giống biển số bằng OpenCV (bổ sung cho trường hợp chữ quá nhỏ / mờ)
# --------------------------------------------------------------------------- #
def count_chars(patch):
    g = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    if h < 40:
        g = cv2.resize(g, None, fx=40.0 / h, fy=40.0 / h)
        h, w = g.shape
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, m = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, st, _ = cv2.connectedComponentsWithStats(m)
    k = 0
    for i in range(1, n):
        x, y, bw, bh, a = st[i]
        if (0.15 * h <= bh <= 0.85 * h and bw <= bh * 1.2 and a >= 0.2 * bw * bh
                and x > 1 and y > 1 and x + bw < w - 1 and y + bh < h - 1):
            k += 1
    return k


def plates_opencv(img, max_frac=0.5):
    H, W = img.shape[:2]
    scale = 1000.0 / max(H, W) if max(H, W) > 1000 else 1.0
    small = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale != 1.0 else img
    sh, sw = small.shape[:2]
    rects = []
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for vmin in (150, 170, 190):  # vùng sáng, ít màu: biển nền trắng
        m = cv2.morphologyEx(cv2.inRange(hsv, (0, 0, vmin), (180, 70, 255)), cv2.MORPH_CLOSE, kern)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        rects += [(cv2.minAreaRect(c), cv2.contourArea(c)) for c in cnts]
    gray = cv2.bilateralFilter(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 11, 17, 17)
    edges = cv2.dilate(cv2.Canny(gray, 30, 200), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:40]:  # tứ giác: biển vàng, xanh...
        approx = cv2.approxPolyDP(c, 0.03 * cv2.arcLength(c, True), True)
        if len(approx) == 4:
            rects.append((cv2.minAreaRect(approx), cv2.contourArea(approx)))

    found = []
    for (center, (w, h), ang), area in rects:
        if min(w, h) < 12:
            continue
        ratio, fill, frac = max(w, h) / min(w, h), area / (w * h), w * h / float(sh * sw)
        if not (2.3 <= ratio <= 6.0 or 1.1 <= ratio <= 1.8) or fill < 0.6 or not 0.0015 <= frac <= max_frac:
            continue
        quad = cv2.boxPoints((center, (w, h), ang)) / scale
        patch = warp_quad(img, quad)
        if patch is not None:
            n = count_chars(patch)
            if n >= 4:
                found.append((n, quad, patch))
    found.sort(key=lambda t: -t[0])
    keep = []
    for n, quad, patch in found:
        bb = bbox(quad)
        if all(iou(bb, k[0]) < 0.4 for k in keep):
            keep.append((bb, dict(quad=order_pts(quad), crop=patch)))
    return [d for _, d in keep[:6]]


# --------------------------------------------------------------------------- #
#  SORT - Simple Online and Realtime Tracking (theo dõi biển số qua các khung hình)
#  Bộ lọc Kalman cho từng biển + ghép cặp bằng IoU (thuật toán Hungarian)
# --------------------------------------------------------------------------- #
def _to_z(b):
    w, h = b[2] - b[0], b[3] - b[1]
    return np.array([[b[0] + w / 2.0], [b[1] + h / 2.0], [w * h], [w / max(h, 1e-6)]], dtype=float)


def _to_box(x):
    s, r = max(float(x[2, 0]), 1e-6), max(float(x[3, 0]), 1e-6)
    w = math.sqrt(s * r)
    h = s / w
    return (float(x[0, 0]) - w / 2.0, float(x[1, 0]) - h / 2.0, float(x[0, 0]) + w / 2.0, float(x[1, 0]) + h / 2.0)


class _Track:
    def __init__(self, tid, box):
        self.id = tid
        self.F = np.eye(7)
        self.F[0, 4] = self.F[1, 5] = self.F[2, 6] = 1.0
        self.H = np.eye(4, 7)
        self.R = np.diag([1, 1, 10, 10]).astype(float)
        self.P = np.diag([10, 10, 10, 10, 10000, 10000, 10000]).astype(float)
        self.Q = np.diag([1, 1, 1, 1, 0.01, 0.01, 0.0001]).astype(float)
        self.x = np.zeros((7, 1))
        self.x[:4] = _to_z(box)
        self.miss = 0

    def predict(self):
        if self.x[6, 0] + self.x[2, 0] <= 0:
            self.x[6, 0] = 0.0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.miss += 1
        return _to_box(self.x)

    def update(self, box):
        self.miss = 0
        y = _to_z(box) - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P


class Sort:
    def __init__(self, max_age=6, iou_thr=0.2):
        self.tracks, self.max_age, self.iou_thr, self.next_id = [], max_age, iou_thr, 1

    def update(self, boxes):
        """boxes: list (x1, y1, x2, y2). Trả về list ID theo đúng thứ tự boxes (cùng 1 biển -> cùng ID)."""
        preds = [t.predict() for t in self.tracks]
        ids = [None] * len(boxes)
        pairs = []
        if preds and boxes:
            M = np.array([[iou(p, b) for b in boxes] for p in preds])
            M = np.nan_to_num(M)
            try:
                from scipy.optimize import linear_sum_assignment
                r, c = linear_sum_assignment(-M)
                pairs = list(zip(r, c))
            except ImportError:  # không có scipy: ghép tham lam theo IoU giảm dần
                order = sorted(((M[i, j], i, j) for i in range(M.shape[0]) for j in range(M.shape[1])), reverse=True)
                ur, uc = set(), set()
                for _, i, j in order:
                    if i not in ur and j not in uc:
                        pairs.append((i, j))
                        ur.add(i)
                        uc.add(j)
            pairs = [(i, j) for i, j in pairs if M[i, j] >= self.iou_thr]
        done = set()
        for i, j in pairs:
            self.tracks[i].update(boxes[j])
            ids[j] = self.tracks[i].id
            done.add(j)
        for j, b in enumerate(boxes):
            if j not in done:
                t = _Track(self.next_id, b)
                self.next_id += 1
                self.tracks.append(t)
                ids[j] = t.id
        self.tracks = [t for t in self.tracks if t.miss <= self.max_age]
        return ids


# --------------------------------------------------------------------------- #
#  Ghép các ký tự do mô hình 2 phát hiện thành chuỗi biển số (1 dòng hoặc 2 dòng, kể cả biển bị nghiêng)
# --------------------------------------------------------------------------- #
def nms_agnostic(dets, thr=0.6):
    keep = []
    for d in sorted(dets, key=lambda d: -d["conf"]):
        if all(iou(d["box"], k["box"]) < thr for k in keep):
            keep.append(d)
    return keep


def _two_means(ys):
    """ys đã sắp xếp. Chia thành 2 nhóm sao cho tổng bình phương sai lệch nhỏ nhất -> (sse, vị trí cắt)."""
    best = (float("inf"), 1)
    for s in range(1, len(ys)):
        a, b = ys[:s], ys[s:]
        sse = float(((a - a.mean()) ** 2).sum() + ((b - b.mean()) ** 2).sum())
        if sse < best[0]:
            best = (sse, s)
    return best


def assemble_chars(dets):
    """dets: list dict(box=(x1,y1,x2,y2), conf, ch). Trả về (text, kind, valid, conf) hoặc None."""
    dets = nms_agnostic(dets)
    if len(dets) < 5:
        return None
    C = np.array([[(d["box"][0] + d["box"][2]) / 2.0, (d["box"][1] + d["box"][3]) / 2.0] for d in dets], float)
    mh = float(np.median([d["box"][3] - d["box"][1] for d in dets]))
    best = None
    for th in np.arange(-85.0, 85.1, 2.5):  # tìm góc nghiêng của hàng chữ
        r = math.radians(th)
        yp = -C[:, 0] * math.sin(r) + C[:, 1] * math.cos(r)
        order = np.argsort(yp)
        sse, cut = _two_means(yp[order])
        if best is None or sse < best[0]:
            best = (sse, th, yp, order, cut)
    _, th, yp, order, cut = best
    r = math.radians(th)
    xp = C[:, 0] * math.cos(r) + C[:, 1] * math.sin(r)
    top, bot = order[:cut], order[cut:]
    two_line = (yp[bot].mean() - yp[top].mean()) > 0.6 * mh and len(top) >= 2 and len(bot) >= 3

    def row(idx):
        idx = sorted(idx, key=lambda i: xp[i])
        return "".join(dets[i]["ch"] for i in idx), float(np.mean([dets[i]["conf"] for i in idx]))

    conf = float(np.mean([d["conf"] for d in dets]))
    if two_line:
        t1, _ = row(top)
        t2, _ = row(bot)
        hd, tl = parse_head(clean(t1)), parse_tail(clean(t2))
        if hd and tl:
            return fmt_two(hd, tl), "2 dòng", True, conf
        raw = clean(t1 + t2)
    else:
        raw, _ = row(range(len(dets)))
        raw = clean(raw)
    p = parse_single(raw)
    if p:
        return fmt_single(*p), "1 dòng", True, conf
    return raw, ("2 dòng" if two_line else "1 dòng"), False, conf


# --------------------------------------------------------------------------- #
#  Tiền xử lý bằng OpenCV: điều chỉnh sáng, khử nhiễu, nắn thẳng biển bị nghiêng
# --------------------------------------------------------------------------- #
def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    return cv2.fastNlMeansDenoisingColored(out, None, 5, 5, 7, 21)


def deskew(crop):
    """Biển nằm nghiêng trong khung cắt -> nắn thẳng. Trả về None nếu không tìm được biển rõ ràng."""
    h, w = crop.shape[:2]
    p = max(6, int(0.15 * max(h, w)))
    pad = cv2.copyMakeBorder(crop, p, p, p, p, cv2.BORDER_CONSTANT, value=(60, 60, 60))
    cands = plates_opencv(pad, max_frac=0.98)
    if not cands:
        return None
    best = max(cands, key=lambda d: cv2.contourArea(d["quad"].astype(np.float32)))
    if cv2.contourArea(best["quad"].astype(np.float32)) < 0.4 * h * w:
        return None
    return best["crop"]


# --------------------------------------------------------------------------- #
#  Hai mô hình YOLOv8 như báo cáo: (1) vùng biển số, (2) ký tự
# --------------------------------------------------------------------------- #
class YoloPipeline:
    def __init__(self, plate_weights, char_weights, conf_plate=0.25, conf_char=0.25):
        from ultralytics import YOLO

        self.model_detect = YOLO(plate_weights)      # tên theo Hình 3.6 của báo cáo
        self.model_char = YOLO(char_weights)
        self.cp, self.cc = conf_plate, conf_char
        names = getattr(self.model_char, "names", {})
        if len(names) != 36:
            print(f"Cảnh báo: mô hình ký tự có {len(names)} lớp (báo cáo dùng 36 lớp: 0-9 và A-Z).")

    # -- Mô hình 1: phát hiện vùng biển số --------------------------------- #
    def detect_plates(self, img):
        imgsz = 1280 if max(img.shape[:2]) > 1280 else 640
        r = self.model_detect(img, conf=self.cp, imgsz=imgsz, verbose=False)[0]
        H, W = img.shape[:2]
        out = []
        for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            x1, y1, x2, y2 = int(max(0, b[0])), int(max(0, b[1])), int(min(W, b[2])), int(min(H, b[3]))
            if x2 - x1 > 4 and y2 - y1 > 4:
                out.append(dict(box=(x1, y1, x2, y2), conf=float(c)))
        return out

    @staticmethod
    def crop(img, box, pad=0.05):
        """Cắt biển số theo tọa độ xyxy, giống Hình 3.6 của báo cáo (có chừa lề nhỏ):
            coordinate = [x_min, y_min, x_max, y_max]
            LpRegion = img[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]"""
        H, W = img.shape[:2]
        x1, y1, x2, y2 = box
        dx, dy = int((x2 - x1) * pad), int((y2 - y1) * pad)
        coordinate = [max(0, x1 - dx), max(0, y1 - dy), min(W, x2 + dx), min(H, y2 + dy)]
        LpRegion = img[coordinate[1]:coordinate[3], coordinate[0]:coordinate[2]]
        return LpRegion

    # -- Mô hình 2: phát hiện ký tự ---------------------------------------- #
    def read_chars(self, crop):
        r = self.model_char(crop, conf=self.cc, imgsz=640, verbose=False, agnostic_nms=True)[0]
        names = self.model_char.names
        dets = [dict(box=tuple(b), conf=float(c), ch=str(names[int(k)]).upper())
                for b, c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist())]
        return assemble_chars(dets)

    def read_plate(self, img, box):
        """Thử lần lượt: ảnh cắt gốc -> ảnh đã nắn thẳng -> ảnh đã tăng sáng/khử nhiễu. Dừng khi đọc ra biển hợp lệ."""
        crop = self.crop(img, box)
        if crop.size == 0:
            return None
        best = None
        for make in (lambda: crop, lambda: deskew(crop), lambda: enhance(crop)):
            v = make()
            if v is None:
                continue
            res = self.read_chars(v)
            if res is None:
                continue
            text, kind, valid, conf = res
            cand = dict(text=text, kind=kind, valid=valid, conf=conf, crop=v)
            if valid:
                return cand
            if best is None or conf > best["conf"]:
                best = cand
        return best


def load_yolo(plate_w, char_w):
    here = os.path.dirname(os.path.abspath(__file__))
    cands = lambda name, given: [given] if given else [os.path.join(os.getcwd(), name), os.path.join(here, name)]
    pw = next((p for p in cands("plate.pt", plate_w) if p and os.path.isfile(p)), None)
    cw = next((p for p in cands("char.pt", char_w) if p and os.path.isfile(p)), None)
    if not (pw and cw):
        return None
    try:
        return YoloPipeline(pw, cw)
    except ImportError:
        print("Có plate.pt và char.pt nhưng chưa cài ultralytics. Chạy: python -m pip install ultralytics")
        return None


def load_engine(required=False):
    try:
        from rapidocr_onnxruntime import RapidOCR
    except ImportError:
        if required:
            sys.exit("Chưa cài RapidOCR. Hãy chạy:\n  python -m pip install rapidocr-onnxruntime")
        return None
    return RapidOCR()


# --------------------------------------------------------------------------- #
#  Phân tích 1 ảnh (hoặc 1 khung hình)
# --------------------------------------------------------------------------- #
def ocr_crop(engine, crop):
    pad = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
    found = lines_to_plates(read_lines(engine, pad))
    return max(found, key=lambda p: p["conf"]) if found else None


def analyze_image(img, yolo=None, engine=None, mode="auto"):
    """Trả về list biển số: dict(quad, text, conf, kind, valid, source, crop)."""
    plates = []
    if yolo is not None and mode in ("auto", "yolo"):
        for det in yolo.detect_plates(img):
            res = yolo.read_plate(img, det["box"])
            source = "yolo"
            if res is None and engine is not None and mode == "auto":  # mô hình ký tự không thấy gì -> thử OCR
                crop = yolo.crop(img, det["box"])
                alt = ocr_crop(engine, crop) if crop.size else None
                if alt:
                    res, source = dict(text=alt["text"], kind=alt["kind"], valid=True, conf=alt["conf"], crop=crop), "ocr"
            if res is None:
                continue
            x1, y1, x2, y2 = det["box"]
            p = dict(quad=np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32), source=source, **res)
            if not p["valid"] and engine is not None and mode == "auto":  # YOLO chưa chắc -> thử đọc lại bằng OCR
                alt = ocr_crop(engine, p["crop"])
                if alt:
                    p.update(text=alt["text"], conf=alt["conf"], kind=alt["kind"], valid=True, source="ocr")
            plates.append(p)
    if engine is not None and mode in ("auto", "ocr"):
        if mode == "ocr" or not any(p["valid"] for p in plates):
            for p in analyze(engine, img):
                p.update(valid=True, source="ocr")
                plates.append(p)
    plates = dedupe(plates)
    for p in plates:
        if p.get("crop") is None:
            p["crop"] = warp_quad(img, p["quad"])
    return plates


# --------------------------------------------------------------------------- #
#  Video / camera: phát hiện -> SORT gán ID -> bình chọn kết quả cho từng biển
# --------------------------------------------------------------------------- #
class VideoProcessor:
    def __init__(self, yolo, engine, mode, skip):
        self.yolo, self.engine, self.mode, self.skip = yolo, engine, mode, max(1, skip)
        self.tracker = Sort()
        self.votes = defaultdict(Counter)   # ID -> {chuỗi biển số: tổng độ tin cậy}
        self.hits = Counter()               # ID -> số khung hình đã thấy
        self.crops = {}
        self.n = 0
        self.last = []

    def step(self, frame):
        self.n += 1
        if (self.n - 1) % self.skip == 0:
            plates = analyze_image(frame, self.yolo, self.engine, self.mode)
            ids = self.tracker.update([bbox(p["quad"]) for p in plates])
            for p, tid in zip(plates, ids):
                p["id"] = tid
                self.hits[tid] += 1
                if p.get("valid", True):
                    self.votes[tid][p["text"]] += p["conf"]
                    if p["conf"] >= self.crops.get(tid, (0, None))[0]:
                        self.crops[tid] = (p["conf"], p["crop"])
            self.last = plates
        for p in self.last:
            v = self.votes.get(p["id"])
            p["final"] = v.most_common(1)[0][0] if v else p["text"]
            if v:
                p["valid"] = True
        return draw_report_video(frame, self.last), self.last

    def final_results(self):
        """Biển số cuối cùng sau khi bình chọn (dạng 59G188888), bỏ các ID chỉ lóe lên 1 lần khi video đủ dài."""
        min_hits = 2 if self.n >= 10 * self.skip else 1
        out = [compact(v.most_common(1)[0][0]) for tid, v in sorted(self.votes.items()) if self.hits[tid] >= min_hits]
        return list(dict.fromkeys(out))

    def summary(self):
        """[(ID, biển số được bình chọn, tổng điểm, số lần thấy)] - chỉ giữ biển thấy từ 2 lần trở lên nếu có nhiều biển."""
        rows = []
        for tid, v in self.votes.items():
            text, score = v.most_common(1)[0]
            rows.append((tid, text, score, self.hits[tid]))
        return sorted(rows)


# --------------------------------------------------------------------------- #
#  Vẽ kết quả, lưu file
# --------------------------------------------------------------------------- #
FONT = cv2.FONT_HERSHEY_SIMPLEX


def compact(text):
    """Cách hiển thị trong báo cáo: 59-G1 888.88 -> 59G188888."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def fit_window(img, lo=0, hi=900):
    """Thu/phóng ảnh cho vừa cửa sổ hiển thị."""
    m = max(img.shape[:2])
    f = hi / float(m) if m > hi else (lo / float(m) if lo and m < lo else 1.0)
    if f == 1.0:
        return img
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)


def draw_report_image(img, plates):
    """Cửa sổ "Detected License Plates" (Hình 3.15): khung xanh lá quanh từng biển số."""
    vis = img.copy()
    th = max(2, int(max(vis.shape[:2]) / 300))
    for p in plates:
        cv2.polylines(vis, [p["quad"].astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), th)
    return vis


def draw_report_video(frame, plates):
    """Cửa sổ "Webcam Stream" (Hình 3.17): khung xanh dương, "ID: n" xanh dương, biển số xanh lá."""
    vis = frame.copy()
    fs = max(0.7, max(vis.shape[:2]) / 1000.0)
    for p in plates:
        q = p["quad"].astype(np.int32)
        cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, (255, 0, 0), 2)
        x, y = int(q[:, 0].min()), int(q[:, 1].min())
        cv2.putText(vis, f"ID: {p.get('id', '')}", (x, max(30, y - int(36 * fs))), FONT, fs, (255, 0, 0), 2)
        cv2.putText(vis, compact(p.get("final") or p["text"]), (x, max(58, y - 6)), FONT, fs, (0, 255, 0), 2)
    return vis


def draw_plates(img, plates):
    vis = img.copy()
    H, W = vis.shape[:2]
    fs = max(0.6, max(H, W) / 1100.0)
    th = max(2, int(max(H, W) / 350))
    for i, p in enumerate(plates, 1):
        ok = p.get("valid", True)
        color = (0, 200, 0) if ok else (0, 140, 255)
        q = p["quad"].astype(np.int32)
        cv2.polylines(vis, [q.reshape(-1, 1, 2)], True, color, th)
        text = p.get("final") or p["text"]
        label = (f"ID{p['id']} " if "id" in p else f"#{i} ") + compact(text) + ("" if ok else " ?")
        x, y = int(q[:, 0].min()), int(q[:, 1].min())
        (tw, tht), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        y = max(tht + 8, y - 8)
        cv2.rectangle(vis, (x, y - tht - 6), (x + tw + 6, y + base), color, -1)
        cv2.putText(vis, label, (x + 3, y - 2), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    return vis


def collect_images(paths):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(IMG_EXT)]
        elif os.path.isfile(p):
            files.append(p)
        else:
            print("Không tìm thấy:", p)
    return files


def process_file(path, out_dir, yolo, engine, mode):
    img = imread(path)
    if img is None:
        return None, []
    plates = analyze_image(img, yolo, engine, mode)
    vis = draw_plates(img, plates)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    imwrite(os.path.join(out_dir, f"{base}_ketqua.jpg"), vis)
    for i, p in enumerate(plates, 1):
        if p.get("crop") is not None:
            imwrite(os.path.join(out_dir, f"{base}_bienso_{i}.jpg"), p["crop"])
    return vis, plates


def csv_row(name, i, p):
    return [name, i, compact(p["text"]), p["text"], f"{p['conf']:.2f}", p["kind"], "co" if p.get("valid", True) else "chua_chac", p.get("source", "")]


def write_csv(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "ket_qua.csv")
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["anh", "stt", "bien_so", "dinh_dang", "do_tin_cay", "loai", "hop_le", "nguon"])
        w.writerows(rows)
    return path


def run_cli(paths, out_dir, yolo, engine, mode):
    files = collect_images(paths)
    if not files:
        print("Không có ảnh nào để xử lý.")
        return
    rows = []
    for f in files:
        _, plates = process_file(f, out_dir, yolo, engine, mode)
        name = os.path.basename(f)
        if not plates:
            print(f"{name}: không tìm thấy biển số")
        for i, p in enumerate(plates, 1):
            flag = "" if p.get("valid", True) else "  [chưa chắc chắn]"
            print(f"{name}: #{i}  {compact(p['text'])}  [{p['text']}]   ({p['conf']:.0%}, biển {p['kind']}, {p.get('source', '')}){flag}")
            rows.append(csv_row(name, i, p))
    print("Đã lưu kết quả vào:", os.path.abspath(out_dir), "| bảng:", write_csv(rows, out_dir))


def run_video_cli(path, out_dir, yolo, engine, mode, skip):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit("Không mở được video: " + path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, os.path.splitext(os.path.basename(path))[0] + "_ketqua.mp4")
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    vp = VideoProcessor(yolo, engine, mode, skip)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        vis, _ = vp.step(frame)
        writer.write(vis)
    cap.release()
    writer.release()
    print("Đã lưu video kết quả:", os.path.abspath(out_path))
    rows = []
    for tid, text, score, hits in vp.summary():
        print(f"ID{tid}: {compact(text)}  [{text}]   (thấy {hits} lần, tổng điểm {score:.2f})")
        rows.append([os.path.basename(path), tid, compact(text), text, f"{score:.2f}", "", "co", ""])
    if not rows:
        print("Không tìm thấy biển số nào trong video.")
    write_csv(rows, out_dir)


# --------------------------------------------------------------------------- #
#  Giao diện đúng như báo cáo (mục 3.2.2, Hình 3.14 - 3.18)
#    Hình 3.14  cửa sổ "Nhận diện biển số xe" có 3 nút
#    Hình 3.15  cửa sổ "Detected License Plates"  (ảnh có khung xanh lá quanh biển số)
#    Hình 3.16  cửa sổ "License Plate Crop"        (biển số đã cắt)
#    Hình 3.17  cửa sổ "Webcam Stream"             (khung xanh dương + "ID: n" + biển số xanh lá)
#    Hình 3.18  hộp thoại "Kết quả": "Biển số nhận diện được: ..., ..."
# --------------------------------------------------------------------------- #
WIN_IMAGE, WIN_CROP, WIN_WEBCAM, WIN_VIDEO = "Detected License Plates", "License Plate Crop", "Webcam Stream", "Video Stream"


def wait_close(win):
    """Giữ cửa sổ OpenCV cho tới khi bấm phím bất kỳ hoặc bấm nút đóng cửa sổ."""
    while cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) >= 1:
        if cv2.waitKey(100) != -1:
            break


def run_gui(yolo, engine, mode, skip):
    import tkinter as tk
    from tkinter import filedialog, messagebox

    def tai_anh():
        path = filedialog.askopenfilename(title="Chọn ảnh", filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff")])
        if not path:
            return
        img = imread(path)
        if img is None:
            messagebox.showerror("Lỗi", "Không đọc được ảnh này.")
            return
        plates = analyze_image(img, yolo, engine, mode)
        if not plates:
            messagebox.showinfo("Kết quả", "Không tìm thấy biển số trong ảnh.")
            return
        cv2.imshow(WIN_IMAGE, fit_window(draw_report_image(img, plates)))
        for k, p in enumerate(plates[:4]):
            if p.get("crop") is not None:
                cv2.imshow(WIN_CROP if k == 0 else f"{WIN_CROP} {k + 1}", fit_window(p["crop"], lo=320, hi=700))
        wait_close(WIN_IMAGE)
        cv2.destroyAllWindows()
        messagebox.showinfo("Kết quả", "Biển số nhận diện được: " + ", ".join(compact(p["text"]) for p in plates))

    def chay_luong(src, win):
        if os.name == "nt" and src == 0:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được video / camera.")
            return
        vp = VideoProcessor(yolo, engine, mode, skip)
        while True:  # bấm q hoặc Esc để dừng
            ok, frame = cap.read()
            if not ok:
                break
            vis, _ = vp.step(frame)
            cv2.imshow(win, fit_window(vis, hi=1000))
            if (cv2.waitKey(1) & 0xFF) in (27, ord("q")) or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
                break
        cap.release()
        cv2.destroyAllWindows()
        res = vp.final_results()
        messagebox.showinfo("Kết quả", ("Biển số nhận diện được: " + ", ".join(res)) if res else "Không tìm thấy biển số.")

    def tai_video():
        path = filedialog.askopenfilename(title="Chọn video", filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv")])
        if path:
            chay_luong(path, WIN_VIDEO)

    root = tk.Tk()
    root.title("Nhận diện biển số xe")
    root.geometry("440x250")
    for text, cmd in (("Tải video để nhận diện biển số", tai_video),
                      ("Tải ảnh để nhận diện biển số", tai_anh),
                      ("Sử dụng Webcam để nhận diện", lambda: chay_luong(0, WIN_WEBCAM))):
        tk.Button(root, text=text, command=cmd, padx=8, pady=3).pack(pady=18)
    root.mainloop()


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="Nhận diện biển số xe (YOLOv8 + OpenCV + SORT) theo báo cáo")
    ap.add_argument("paths", nargs="*", help="ảnh hoặc thư mục ảnh (bỏ trống = mở giao diện)")
    ap.add_argument("--video", help="xử lý 1 file video, lưu video kết quả")
    ap.add_argument("--gui", action="store_true", help="mở giao diện 3 nút như báo cáo")
    ap.add_argument("--mode", choices=["auto", "yolo", "ocr"], default="auto", help="xem phần mô tả đầu file")
    ap.add_argument("--plate-weights", help="trọng số mô hình biển số (mặc định: plate.pt)")
    ap.add_argument("--char-weights", help="trọng số mô hình ký tự (mặc định: char.pt)")
    ap.add_argument("--skip", type=int, default=0, help="video: cứ N khung hình mới nhận diện 1 lần")
    ap.add_argument("-o", "--out", default=OUT_DIR, help="thư mục lưu kết quả (mặc định: ket_qua)")
    a = ap.parse_args()

    yolo = load_yolo(a.plate_weights, a.char_weights) if a.mode != "ocr" else None
    if a.mode == "yolo" and yolo is None:
        sys.exit("Chế độ yolo cần plate.pt và char.pt (xem huan_luyen_yolov8_colab.ipynb) và thư viện ultralytics.")
    engine = load_engine(required=(yolo is None)) if a.mode != "yolo" else None
    mode = a.mode
    print("Chế độ:", "YOLOv8 theo báo cáo" + (" + OCR dự phòng" if engine and mode == "auto" else "") if yolo
          else "OCR (chưa có plate.pt/char.pt nên chưa dùng YOLOv8)")
    skip = a.skip or (2 if yolo else 8)

    if a.video:
        run_video_cli(a.video, a.out, yolo, engine, mode, skip)
    elif a.paths and not a.gui:
        run_cli(a.paths, a.out, yolo, engine, mode)
    else:
        run_gui(yolo, engine, mode, skip)


if __name__ == "__main__":
    main()
