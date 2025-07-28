# **Our Solution: Problem 1A**

### **What This Is**

This document explains how our Universal Document Structure Extractor works. We built this tool for the **Round 1A: Understand Your Document** challenge. We had to create a tool that could figure out the structure of any PDF just by looking at its layout. Our final solution came after many tries. We moved from simple keyword searching to a smarter, multi-step process that decodes a document's structure with high accuracy and meets all the tough performance rules.

---

## **1. The Goal: Smart Analysis with Strict Rules**

The main goal was to build a tool that can take any PDF and pull out its title and a structured outline (H1, H2, H3) without knowing anything about the document's topic or language. The final output needed to be a clean JSON file.

This was difficult because of the strict rules:

> **Core Rules**

> * **Speed:** Must finish in 10 seconds or less for a 50-page PDF.

> * **Size:** Any models used must be 200MB or smaller.

> * **Runtime:** Must run on a normal CPU (amd64) without a GPU.

> * **Offline:** Cannot use the internet.

The tool had to be fast, small, and work on any document you give it.

---

## **2. Our Main Idea: Look at the Style, Not the Words**

We learned something important from our tests: for finding headings, how text *looks* is more important than what it *says*.

> **"Forget the words. A document's real structure is shown by its font size, bold text, and indentation. Our job is to read that style."**

This main idea let us build a tool that works with any language. We improved it by adding a language detection step, which led to our final design focused on being **universal, accurate, and fast.**

---

## **3. How It Works: A Multi-Step Process**

Every piece of text goes through several checks. Only the text that passes these checks is marked as a title or a heading.

![Our Final Architecture Flowchart](./1a_solution/git_1a.png)

### **Step 0: Figure Out the Language**

Before anything else, we find out what language the document is written in.

* **What it does:** It scans the text to automatically detect the main language.
* **Why we do it:** This helps us prove our tool works for many languages, which was a bonus goal in the challenge.
* **Tool Used:** The fast and small `lid.176.ftz` model.

### **Step 1: Removing Junk Content**

First, we do an aggressive filtering to get rid of all the text that isn't part of the main content.

* **What it does:** It scans each page and throws away common junk like tables, footnotes, and page numbers.
* **Why we do it:** Headings and titles are never in these places. Removing them first makes the next steps more accurate and much faster, helping us meet the **10-second time limit**.
* **Tool Used:** PyMuPDF (fitz).

### **Step 2: Finding the Main Text Style**

This step learns the document's unique style.

* **What it does:** It does a count of all font styles to find the most common one (size, bold, etc.). We call this the "body text."
* **Why we do it:** Once we know what normal text looks like, it's easy to spot anything different. All other styles are then ranked to guess the heading levels (e.g., biggest font is probably H1).
* **Tool Used:** Python's `collections.Counter`.

### **Step 3: Assigning Headings in Two Layers**

This is our trick for making sure the headings are accurate. We check our work.

* **What it does:** It's a two-part process. **Layer 1** makes a quick first guess based on how far the text is from the left margin (indentation). **Layer 2** then checks that all headings at the same level (like all H2s) have the same font style. If one doesn't match, it gets moved to a lower level (like H3).
* **Why we do it:** This two-layer check is very reliable. Using just indentation or just font style makes mistakes. Using both, one after the other, fixes those mistakes.

### **Step 4: Finding the Main Title**

The document's main title gets its own special step.

* **What it does:** This simple step only looks at the first page and finds the text with the biggest font size, marking it as the title.
* **Why we do it:** This is a simple and very reliable way to grab the main title, which was a required part of the final JSON output.

---

## **4. Our Journey: A History of Failed Tries**

We figured out our final design by trying a lot of things that didn't work.

| Attempt | Approach | What Went Wrong | What We Learned |

| :--- | :--- | :--- | :--- |

|**#1**|**Keyword Search**`<br>` Searched for words like "Chapter," "Section," "Introduction." | This almost never worked. Most real documents don't use these generic words in their headings. | Relying on specific words is a bad idea. The tool needs to work without understanding the words. |

|**#2**|**Font Size Only**`<br>` Decided heading levels only based on font size. | This was better but still made mistakes. It often marked big quotes as main headings and couldn't tell H2 and H3 apart if they were the same size. | Font size is a good clue, but you need more information. |

|**#3**|**Indentation Only**`<br>` Used only the text's distance from the left margin to guess the hierarchy. | This couldn't tell the difference between a main heading (H1) and a sub-heading (H2) if they were both on the far left. It also thought bullet points were headings. | Indentation is another good clue, but it's also not enough by itself. |

|**#4**|**The Final Hybrid**`<br>` Our current method. It combines all the clues in a smart, multi-step process. | Our first tries to combine all clues into a single "score" were too complicated and still made mistakes. | The real solution wasn't just to combine the clues, but to use them in the right order: use indentation to make a first draft of the outline, then use font style to double-check and fix it. |

---

## **5. Why Our Solution Works Well**

Our multi-step process is strong, accurate, and works for any document.

* ✅ **Works on Any Document:** Because we check the visual style and not the words, our tool works with PDFs in any language.
* ✅ **Works with Many Languages:** We added a quick language-detection step using the **`lid.176.ftz`** model. This lets us handle languages like **French, German, Spanish, Portuguese, and Russian**, which helped us get the bonus points.
* ✅ **Very Accurate:** The **Two-Layer** heading check is our key idea. It finds and fixes mistakes that simpler tools would miss.
* ✅ **Super Fast & Follows the Rules:** The whole process uses fast math and logic, not slow AI models. This means we easily meet the **10-second time limit** and **200MB size limit**.
* ✅ **Easy to Trust:** Each step is simple and predictable. This makes the tool easy to debug and not a "black box" like some AI solutions.
