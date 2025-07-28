#!/usr/bin/env python3
"""
Universal PDF Title and Headings Extractor using PyMuPDF

This script extracts titles and headings from PDF files by analyzing document
structure, font styles, and layout, without relying on hardcoded keywords.
It includes advanced checks for headers/footers, with a special exception
for the header area on page 1 to correctly capture the document title.
"""

import json
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Set
import fitz  # PyMuPDF
from collections import Counter, defaultdict
import fasttext

class StructuralPdfExtractor:
    """
    Extracts document structure based on visual and layout analysis.
    """
    def __init__(self, lang_model_path: str):
        """Initializes the extractor and loads the language model."""
        self.doc_styles = {}
        self.body_style = {}
        self.header_footer_fragments: Set[str] = set()
        self.page_heights: Dict[int, float] = {}
        
        try:
            self.lang_model = fasttext.load_model(lang_model_path)
            print(f"✓ Language model loaded successfully from: {lang_model_path}")
        except ValueError as e:
            print(f"Error loading language model: {e}", file=sys.stderr)
            print("Please ensure you have downloaded the 'lid.176.ftz' model and provided the correct path.", file=sys.stderr)
            sys.exit(1)

    def _get_style_key(self, span: Dict) -> tuple:
        """Creates a unique identifier for a text style."""
        return (
            span['font'],
            round(span['size']),
            span.get('color', 0),
            "bold" in span['font'].lower(),
            "italic" in span['font'].lower()
        )

    def _identify_recurring_headers_footers(self, doc: fitz.Document):
        """
        Finds text fragments that recur across multiple pages in header/footer zones
        to build a rejection list.
        """
        candidates = defaultdict(list)
        for page_num, page in enumerate(doc):
            header_zone = fitz.Rect(0, 0, page.rect.width, page.rect.height * 0.12)
            footer_zone = fitz.Rect(0, page.rect.height * 0.88, page.rect.width, page.rect.height)

            for zone in [header_zone, footer_zone]:
                text = page.get_text("text", clip=zone).strip()
                if text:
                    key = " ".join(text.lower().split())
                    if key:
                        candidates[key].append(page_num)

        num_pages = len(doc)
        for text, pages in candidates.items():
            if len(set(pages)) > 2 or (num_pages > 3 and len(set(pages)) > num_pages * 0.3):
                self.header_footer_fragments.add(text)

    def _is_header_or_footer(self, block: Dict, page_height: float) -> bool:
        """
        A strong check to determine if a block is a header or footer.
        """
        block_bbox = fitz.Rect(block['bbox'])
        text_content = " ".join(" ".join(s['text'].lower().split()) for s in block['spans'])
        first_span = block['spans'][0] if block['spans'] else {}

        if text_content in self.header_footer_fragments:
            return True

        in_header_zone = block_bbox.y0 < page_height * 0.12
        in_footer_zone = block_bbox.y1 > page_height * 0.88

        if in_header_zone or in_footer_zone:
            if self.body_style and first_span.get('size', 0) <= self.body_style.get('size', 0) + 0.5:
                return True
            if re.fullmatch(r'page \d+|\d+', text_content):
                return True
        return False

    def _clean_heading_text(self, text: str) -> str:
        """
        Removes long sequences of decorative characters from the end of a heading.
        """
        return re.sub(r'\s*([._\-—:•\s]){3,}\s*$', '', text).strip()

    def _analyze_document_styles(self, blocks: List[Dict]):
        """
        Analyzes all text blocks to find the body text style and create a
        hierarchy of heading styles.
        """
        if not blocks: return

        style_counts = Counter(self._get_style_key(span) for block in blocks for span in block['spans'])
        if not style_counts: return

        self.body_style_key = style_counts.most_common(1)[0][0]
        self.body_style = {
            'font': self.body_style_key[0], 'size': self.body_style_key[1], 'color': self.body_style_key[2],
            'bold': self.body_style_key[3], 'italic': self.body_style_key[4]
        }

        potential_heading_styles = [
            key for key, count in style_counts.items() if key != self.body_style_key
        ]

        ranked_styles = sorted(
            potential_heading_styles, key=lambda key: (-key[1], -int(key[3])), reverse=False
        )
        self.style_map = {style_key: f"H{i+1}" for i, style_key in enumerate(ranked_styles)}

    def _extract_blocks_from_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extracts all text blocks, filtering out tables and footers separated by lines.
        """
        doc = fitz.open(pdf_path)
        # Store page heights for later use in positional checks
        self.page_heights = {p.number + 1: p.rect.height for p in doc}
        self._identify_recurring_headers_footers(doc)
        
        all_blocks = []
        for page in doc:
            page_height = page.rect.height
            
            footer_line_y = None
            footer_zone_top = page.rect.height * 0.80 
            drawings = page.get_drawings()
            horizontal_lines = []

            for path in drawings:
                for item in path["items"]:
                    if item[0] == "l":
                        p1, p2 = item[1], item[2]
                        if abs(p1.y - p2.y) < 1 and p1.y > footer_zone_top:
                            horizontal_lines.append(p1.y)

            if horizontal_lines:
                footer_line_y = min(horizontal_lines)

            try:
                table_bboxes = [table.bbox for table in page.find_tables()]
            except Exception:
                table_bboxes = []

            blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_DICT)["blocks"]
            for block in blocks:
                if block.get('type') != 0 or not block.get('lines'): continue

                block_rect = fitz.Rect(block['bbox'])
                
                if footer_line_y and block_rect.y0 > footer_line_y:
                    continue
                
                is_in_table = False
                for table_bbox in table_bboxes:
                    intersect_rect = block_rect & table_bbox
                    if intersect_rect.is_empty: continue
                    block_area = block_rect.get_area()
                    intersect_area = intersect_rect.get_area()
                    if block_area > 0 and (intersect_area / block_area > 0.5):
                        is_in_table = True
                        break
                if is_in_table: continue

                temp_block_info = {
                    'bbox': block['bbox'],
                    'spans': [s for line in block['lines'] for s in line['spans'] if s['text'].strip()]
                }
                if not temp_block_info['spans']: continue
                if self._is_header_or_footer(temp_block_info, page_height): continue

                spans_with_style = []
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span['text'].strip():
                            span['bold'] = "bold" in span['font'].lower()
                            span['italic'] = "italic" in span['font'].lower()
                            spans_with_style.append(span)
                
                if spans_with_style:
                    all_blocks.append({
                        'bbox': block['bbox'], 'page': page.number + 1, 'spans': spans_with_style
                    })
        doc.close()
        return all_blocks
        
    def _assign_and_refine_heading_levels(self, candidate_headings: List[Dict]) -> List[Dict]:
        """
        Assigns heading levels based on indentation (Layer 1), then refines them
        using font styles (Layer 2), and filters for H1-H4.
        """
        if not candidate_headings: return []

        unique_indents = sorted(list(set(h['x_indent'] for h in candidate_headings)))
        indent_level_map = {}
        if unique_indents:
            level_num = 1
            indent_level_map[unique_indents[0]] = "H1"
            for i in range(1, len(unique_indents)):
                if unique_indents[i] - unique_indents[i-1] > 3.0:
                    level_num += 1
                indent_level_map[unique_indents[i]] = f"H{level_num}"
        for h in candidate_headings:
            h['level'] = indent_level_map.get(h['x_indent'], "H99")

        for i in range(1, 4):
            current_level = f"H{i}"
            next_level = f"H{i + 1}"
            level_headings = [h for h in candidate_headings if h.get('level') == current_level]
            if not level_headings: continue
            style_counter = Counter((round(h['size']), h['bold']) for h in level_headings)
            if not style_counter: continue
            dominant_style = style_counter.most_common(1)[0][0]
            for heading in level_headings:
                heading_style = (round(heading['size']), heading['bold'])
                if heading_style != dominant_style:
                    heading['level'] = next_level

        final_outline = []
        for h in candidate_headings:
            level_num_str = h['level'][1:]
            if level_num_str.isdigit() and int(level_num_str) <= 4:
                final_outline.append({
                    "level": h['level'],
                    "text": h['text'],
                    "page": h['page']
                })
        return final_outline

    def extract_title_and_headings(self, pdf_path: str) -> Dict[str, Any]:
        """
        The main function to extract the title and a structured outline.
        The heading level logic is applied on a per-page basis.
        """
        raw_blocks = self._extract_blocks_from_pdf(pdf_path)
        self._analyze_document_styles(raw_blocks)
        if not self.style_map: return {"title": "", "outline": []}

        # --- NEW TITLE EXTRACTION LOGIC ---
        # Find the biggest text on the entirety of Page 1.
        title = ""
        page_1_blocks = [b for b in raw_blocks if b['page'] == 1]
        
        if page_1_blocks:
            # Find the maximum font size on Page 1
            max_font_size = 0
            for block in page_1_blocks:
                if block['spans']:
                    max_font_size = max(max_font_size, block['spans'][0]['size'])
            
            # Collect all blocks that have this maximum font size
            title_blocks = []
            for block in page_1_blocks:
                if not block['spans']: continue
                # Use a small tolerance for floating point comparisons
                if abs(block['spans'][0]['size'] - max_font_size) < 0.1:
                    title_blocks.append(block)

            # Sort the collected title blocks by their vertical position
            title_blocks.sort(key=lambda b: b['bbox'][1])
            
            # Join the text from these blocks to form the title
            title = " ".join(" ".join(s['text'] for s in block['spans']) for block in title_blocks)
        # --- END NEW TITLE EXTRACTION LOGIC ---
        
        # --- Heading Candidate Gathering ---
        all_candidate_headings = []
        sorted_blocks = sorted(raw_blocks, key=lambda b: (b['page'], b['bbox'][1]))
        
        for block in sorted_blocks:
            full_block_text = " ".join(s['text'] for s in block['spans'])
            # Exclude text that was identified as the title
            if block['page'] == 1 and full_block_text in title: continue
            if not block['spans']: continue

            block_style_key = self._get_style_key(block['spans'][0])
            word_count = len(full_block_text.split())

            if block_style_key in self.style_map and word_count < 50:
                first_span = block['spans'][0]
                
                page_height = self.page_heights.get(block['page'])
                is_in_footer_zone = page_height and (block['bbox'][1] > page_height * 0.85)
                is_smaller = self.body_style and (first_span['size'] < self.body_style['size'])
                
                if is_in_footer_zone and is_smaller:
                    continue

                cleaned_text = self._clean_heading_text(full_block_text)
                if cleaned_text:
                    all_candidate_headings.append({
                        "text": cleaned_text, "page": block['page'], "x_indent": block['bbox'][0],
                        "size": first_span['size'], "bold": "bold" in first_span['font'].lower(),
                    })
        
        # Group candidates by page and process each page independently
        candidates_by_page = defaultdict(list)
        for cand in all_candidate_headings:
            candidates_by_page[cand['page']].append(cand)

        final_outline = []
        for page_num in sorted(candidates_by_page.keys()):
            page_candidates = candidates_by_page[page_num]
            page_outline = self._assign_and_refine_heading_levels(page_candidates)
            final_outline.extend(page_outline)

        return {"title": self._clean_heading_text(title), "outline": final_outline}

# --- MODIFIED MAIN FUNCTION FOR DOCKER CONTAINER ---
def main():
    # Fixed paths for Docker container
    input_path = Path("app/input")
    output_path = Path("app/output")
    lang_model_path = "app/lid.176.ftz"

    if not input_path.is_dir():
        print(f"Error: Input path '{input_path}' is not a valid directory.", file=sys.stderr)
        sys.exit(1)
        
    # Create the output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    extractor = StructuralPdfExtractor(lang_model_path=lang_model_path)

    pdf_files = list(input_path.glob("*.pdf"))
    if not pdf_files:
        print(f"Warning: No PDF files found in '{input_path}'.", file=sys.stderr)
        sys.exit(0)
    
    print(f"\nFound {len(pdf_files)} PDF file(s) to process in '{input_path}'.")
    
    total_processed = 0
    for pdf_file_path in pdf_files:
        # Define the output file path inside the specified output directory
        output_file_path = output_path / (pdf_file_path.stem + ".json")
        
        try:
            print("-" * 60)
            print(f"Processing PDF: {pdf_file_path.name}")
            result = extractor.extract_title_and_headings(str(pdf_file_path))
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
                
            print(f"✓ Extraction completed successfully!")
            print(f"  Output saved to: {output_file_path}")
            print(f"  Title: {result['title']}")
            print(f"  Headings Found: {len(result['outline'])}")
            total_processed += 1
            
        except Exception as e:
            print(f"Error processing file '{pdf_file_path.name}': {e}", file=sys.stderr)
    
    print("-" * 60)
    print(f"\nFinished. Successfully processed {total_processed} out of {len(pdf_files)} files.")


if __name__ == "__main__":
    main()