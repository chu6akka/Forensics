import re
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from tkinter import messagebox, ttk, filedialog

from openpyxl import Workbook
from pymorphy3 import MorphAnalyzer

WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")


@dataclass
class LemmaEntry:
    lemma: str
    count: int
    forms: dict  # form -> pos


class LemmaAnalyzerApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Анализ словоформ")
        self.morph = MorphAnalyzer()
        self.entries: list[LemmaEntry] = []

        self._build_ui()

    def _build_ui(self) -> None:
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.grid(row=0, column=0, sticky="nsew")

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)

        ttk.Label(main_frame, text="Введите текст для анализа:").grid(
            row=0, column=0, sticky="w"
        )

        self.text_input = tk.Text(main_frame, height=8, wrap="word")
        self.text_input.grid(row=1, column=0, sticky="nsew", pady=(5, 10))

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, sticky="ew")
        buttons_frame.columnconfigure(0, weight=1)

        ttk.Button(buttons_frame, text="Проверить", command=self.analyze).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(buttons_frame, text="Экспорт в Excel", command=self.export_excel).grid(
            row=0, column=1, sticky="w", padx=10
        )

        self.tree = ttk.Treeview(
            main_frame,
            columns=("lemma", "count", "forms"),
            show="headings",
            height=12,
        )
        self.tree.heading("lemma", text="Начальная форма")
        self.tree.heading("count", text="Количество словоформ")
        self.tree.heading("forms", text="Словоформы (с частью речи)")
        self.tree.column("lemma", width=200, anchor="w")
        self.tree.column("count", width=160, anchor="center")
        self.tree.column("forms", width=500, anchor="w")
        self.tree.grid(row=3, column=0, sticky="nsew", pady=(10, 0))

        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=3, column=1, sticky="ns", pady=(10, 0))

    def analyze(self) -> None:
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Нет текста", "Введите текст для анализа.")
            return

        lemma_data: dict[str, LemmaEntry] = {}
        lemma_counts: defaultdict[str, int] = defaultdict(int)
        lemma_forms: defaultdict[str, dict[str, str]] = defaultdict(dict)

        for match in WORD_RE.finditer(text):
            token = match.group(0)
            parse = self.morph.parse(token)[0]
            lemma = parse.normal_form
            pos = parse.tag.POS or "Неизвестно"
            form_key = token.lower()

            lemma_counts[lemma] += 1
            if form_key not in lemma_forms[lemma]:
                lemma_forms[lemma][form_key] = pos

        for lemma, count in sorted(lemma_counts.items()):
            forms = lemma_forms[lemma]
            lemma_data[lemma] = LemmaEntry(lemma=lemma, count=count, forms=forms)

        self.entries = list(lemma_data.values())
        self._populate_table()

    def _populate_table(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

        for entry in self.entries:
            forms_list = ", ".join(
                f"{form} ({pos})" for form, pos in sorted(entry.forms.items())
            )
            self.tree.insert("", "end", values=(entry.lemma, entry.count, forms_list))

    def export_excel(self) -> None:
        if not self.entries:
            messagebox.showinfo(
                "Нет данных", "Сначала выполните анализ, чтобы экспортировать результат."
            )
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx")],
            title="Сохранить как",
        )
        if not file_path:
            return

        workbook = Workbook()
        sheet = workbook.active
        sheet.title = "Анализ"

        sheet.append(
            [
                "Начальная форма",
                "Количество словоформ",
                "Словоформы (с частью речи)",
            ]
        )

        for entry in self.entries:
            forms_list = ", ".join(
                f"{form} ({pos})" for form, pos in sorted(entry.forms.items())
            )
            sheet.append([entry.lemma, entry.count, forms_list])

        workbook.save(file_path)
        messagebox.showinfo("Готово", f"Файл сохранён: {file_path}")


if __name__ == "__main__":
    root = tk.Tk()
    app = LemmaAnalyzerApp(root)
    root.mainloop()
