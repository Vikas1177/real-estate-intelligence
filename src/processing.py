import fitz
from PIL import Image, ImageOps, ImageFilter
import pytesseract
import numpy as np
import re
from typing import List
import io

OCR_WORD_THRESHOLD = 35
OCR_ZOOM = 3
USE_BINARIZE = True
CHUNK_WORDS = 400
CHUNK_OVERLAP_WORDS = 80

def pil_from_page(page, zoom: float = OCR_ZOOM):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return img

def preprocess_image_for_ocr(img: Image.Image, binarize: bool = USE_BINARIZE) -> Image.Image:
    img = img.convert("L")
    img = ImageOps.autocontrast(img)
    if binarize:
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
    img = img.filter(ImageFilter.MedianFilter(size=3))
    return img

def ocr_image_get_text_and_confidence(img: Image.Image):
    custom_config = r'--oem 3 --psm 3'
    text = pytesseract.image_to_string(img, lang='eng', config=custom_config)
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, lang='eng', config=custom_config)
    confs = []
    for c in data.get('conf', []):
        try:
            ci = float(c)
            if ci >= 0:
                confs.append(ci)
        except:
            continue
    avg_conf = float(np.mean(confs)) if len(confs) > 0 else 0.0
    return text.strip(), avg_conf

def format_table_to_markdown(table_data: List[List[str]]) -> str:
    if not table_data:
        return ""
    clean_rows = [[cell.replace("\n", " ").strip() if cell else "" for cell in row] for row in table_data]
    if not clean_rows:
        return ""
    header = clean_rows[0]
    md = "| " + " | ".join(header) + " |\n"
    md += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in clean_rows[1:]:
        md += "| " + " | ".join(row) + " |\n"
    return md + "\n"

def is_block_inside_table(block_rect, table_rects, threshold=0.6):
    b_area = (block_rect[2] - block_rect[0]) * (block_rect[3] - block_rect[1])
    if b_area <= 0:
        return False
    b_rect = fitz.Rect(block_rect)
    for t_rect in table_rects:
        inter = b_rect & t_rect
        if inter.get_area() / b_area > threshold:
            return True
    return False

def extract_text_and_tables(page) -> str:
    table_items = []
    table_rects = []
    try:
        tables = page.find_tables()
    except Exception:
        tables = []
    for tab in tables:
        try:
            bbox = tab.bbox
            table_rects.append(fitz.Rect(bbox))
            content = tab.extract()
            md_text = format_table_to_markdown(content)
            table_items.append((bbox[1], md_text))
        except Exception:
            continue
    blocks = page.get_text("blocks")
    text_items = []
    for b in blocks:
        if b[6] != 0:
            continue
        if is_block_inside_table((b[0], b[1], b[2], b[3]), table_rects):
            continue
        text = b[4].strip()
        if not text:
            continue
        text_items.append((b[1], text))
    all_items = table_items + text_items
    all_items.sort(key=lambda x: x[0])
    final_text = "\n\n".join([item[1] for item in all_items])
    return final_text

def extract_text_pages_smart(pdf_path: str, ocr_word_threshold: int = OCR_WORD_THRESHOLD) -> List[dict]:
    doc = fitz.open(pdf_path)
    pages_out = []
    for i, page in enumerate(doc, start=1):
        try:
            page_text = extract_text_and_tables(page)
        except Exception as e:
            page_text = page.get_text("text")
        ocr_used = False
        ocr_conf = 0.0
        if len(page_text.split()) < ocr_word_threshold:
            try:
                img = pil_from_page(page, zoom=OCR_ZOOM)
                img_proc = preprocess_image_for_ocr(img)
                ocr_text, conf = ocr_image_get_text_and_confidence(img_proc)
                if len(ocr_text.split()) > len(page_text.split()):
                    page_text = ocr_text
                    ocr_used = True
                    ocr_conf = conf
            except Exception:
                pass
        pages_out.append({
            "page_num": i,
            "text": page_text,
            "ocr_used": ocr_used,
            "ocr_confidence": float(ocr_conf)
        })
    doc.close()
    return pages_out

_sentence_split_regex = re.compile(r'(?<=[\.\?\!])\s+')

def split_into_sentences(text: str):
    if not text:
        return []
    s = text.replace('\r\n', '\n').replace('\r', '\n')
    s = re.sub(r'\n+', ' ', s)
    parts = _sentence_split_regex.split(s)
    sentences = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        sentences.append(p)
    return sentences

def chunk_text(text: str, pdf_name: str, page_num: int, target_words:int = CHUNK_WORDS, overlap_words:int = CHUNK_OVERLAP_WORDS):
    if not text or not text.strip():
        return []
    sentences = split_into_sentences(text)
    if not sentences:
        words = text.split()
        chunks = []
        i = 0
        chunk_id = 0
        while i < len(words):
            j = min(len(words), i + target_words)
            chunk_words = words[i:j]
            chunks.append({
                "pdf_name": pdf_name,
                "page": page_num,
                "chunk_id": f"{pdf_name}__p{page_num}__c{chunk_id}",
                "text": " ".join(chunk_words)
            })
            chunk_id += 1
            i = j - overlap_words if (j - overlap_words) > i else j
        return chunks
    chunks = []
    current_words = []
    current_len = 0
    chunk_id = 0
    for sent in sentences:
        sent_words = sent.split()
        sent_len = len(sent_words)
        if sent_len >= int(1.5 * target_words):
            if current_words:
                chunks.append({
                    "pdf_name": pdf_name,
                    "page": page_num,
                    "chunk_id": f"{pdf_name}__p{page_num}__c{chunk_id}",
                    "text": " ".join(current_words)
                })
                chunk_id += 1
                tail = " ".join(current_words[-overlap_words:]) if len(current_words) >= overlap_words else " ".join(current_words)
                current_words = tail.split() if tail else []
                current_len = len(current_words)
            chunks.append({
                "pdf_name": pdf_name,
                "page": page_num,
                "chunk_id": f"{pdf_name}__p{page_num}__c{chunk_id}",
                "text": sent
            })
            chunk_id += 1
            current_words = []
            current_len = 0
            continue
        if current_len + sent_len <= int(1.2 * target_words):
            current_words.extend(sent_words)
            current_len += sent_len
        else:
            if current_words:
                chunks.append({
                    "pdf_name": pdf_name,
                    "page": page_num,
                    "chunk_id": f"{pdf_name}__p{page_num}__c{chunk_id}",
                    "text": " ".join(current_words)
                })
                chunk_id += 1
            tail = current_words[-overlap_words:] if len(current_words) >= overlap_words else current_words
            current_words = tail.copy()
            current_len = len(current_words)
            current_words.extend(sent_words)
            current_len += sent_len
    if current_words:
        chunks.append({
            "pdf_name": pdf_name,
            "page": page_num,
            "chunk_id": f"{pdf_name}__p{page_num}__c{chunk_id}",
            "text": " ".join(current_words)
        })
    return chunks