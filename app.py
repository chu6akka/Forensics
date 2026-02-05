import re
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from tkinter import messagebox, ttk, filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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
        self.pos_counts: dict[str, int] = {}
        self.total_words: int = 0

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
        self._add_text_context_menu(self.text_input)

        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=2, column=0, sticky="ew")
        buttons_frame.columnconfigure(0, weight=1)

        ttk.Button(buttons_frame, text="Проверить", command=self.analyze).grid(
            row=0, column=0, sticky="w"
        )
        ttk.Button(buttons_frame, text="Экспорт в Excel", command=self.export_excel).grid(
            row=0, column=1, sticky="w", padx=10
        )
        ttk.Button(
            buttons_frame,
            text="Показать коэффициенты",
            command=self.show_pos_coefficients,
        ).grid(row=0, column=2, sticky="w", padx=10)

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

    def _add_text_context_menu(self, widget: tk.Text) -> None:
        menu = tk.Menu(widget, tearoff=0)
        menu.add_command(label="Вырезать", command=lambda: widget.event_generate("<<Cut>>"))
        menu.add_command(label="Копировать", command=lambda: widget.event_generate("<<Copy>>"))
        menu.add_command(label="Вставить", command=lambda: widget.event_generate("<<Paste>>"))
        menu.add_separator()
        menu.add_command(label="Выделить всё", command=lambda: widget.event_generate("<<SelectAll>>"))

        def show_menu(event: tk.Event) -> None:
            menu.tk_popup(event.x_root, event.y_root)

        widget.bind("<Button-3>", show_menu)
        widget.bind("<Control-v>", lambda event: widget.event_generate("<<Paste>>"))
        widget.bind("<Control-V>", lambda event: widget.event_generate("<<Paste>>"))
        widget.bind("<Control-Shift-V>", lambda event: widget.event_generate("<<Paste>>"))

    def analyze(self) -> None:
        text = self.text_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showinfo("Нет текста", "Введите текст для анализа.")
            return

        lemma_data: dict[str, LemmaEntry] = {}
        lemma_counts: defaultdict[str, int] = defaultdict(int)
        lemma_forms: defaultdict[str, dict[str, str]] = defaultdict(dict)
        pos_counts: defaultdict[str, int] = defaultdict(int)
        total_words = 0

        for match in WORD_RE.finditer(text):
            token = match.group(0)
            parse = self.morph.parse(token)[0]
            lemma = parse.normal_form
            pos = parse.tag.POS or "Неизвестно"
            form_key = token.lower()

            lemma_counts[lemma] += 1
            if form_key not in lemma_forms[lemma]:
                lemma_forms[lemma][form_key] = pos
            pos_counts[pos] += 1
            total_words += 1

        for lemma, count in sorted(lemma_counts.items()):
            forms = lemma_forms[lemma]
            lemma_data[lemma] = LemmaEntry(lemma=lemma, count=count, forms=forms)

        self.entries = list(lemma_data.values())
        self.pos_counts = dict(pos_counts)
        self.total_words = total_words
        self._populate_table()

    def show_pos_coefficients(self) -> None:
        if not self.pos_counts or self.total_words == 0:
            messagebox.showinfo(
                "Нет данных",
                "Сначала выполните анализ, чтобы рассчитать коэффициенты.",
            )
            return

        coefficients = {
            pos: count / self.total_words for pos, count in self.pos_counts.items()
        }

        window = tk.Toplevel(self.root)
        window.title("Частотные коэффициенты частей речи")

        figure, ax = plt.subplots(figsize=(8, 4))
        labels = list(coefficients.keys())
        values = [coefficients[label] for label in labels]

        ax.bar(labels, values, color="#4c78a8")
        ax.set_ylabel("Коэффициент")
        ax.set_xlabel("Часть речи")
        ax.set_title("Частотные коэффициенты частей речи")
        ax.set_ylim(0, max(values) * 1.2 if values else 1)
        ax.tick_params(axis="x", rotation=45)
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

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
