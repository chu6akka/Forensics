import re
import tkinter as tk
from collections import defaultdict
from dataclasses import dataclass
from tkinter import messagebox, ttk, filedialog

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from openpyxl import Workbook
from pymorphy3 import MorphAnalyzer
import stanza
import torch

WORD_RE = re.compile(r"[А-Яа-яЁё]+(?:-[А-Яа-яЁё]+)*")
UD_POS_RU = {
    "ADJ": "Прилагательное",
    "ADP": "Предлог",
    "ADV": "Наречие",
    "AUX": "Вспомогательный глагол",
    "CCONJ": "Сочинительный союз",
    "DET": "Определитель",
    "INTJ": "Междометие",
    "NOUN": "Существительное",
    "NUM": "Числительное",
    "PART": "Частица",
    "PRON": "Местоимение",
    "PROPN": "Имя собственное",
    "SCONJ": "Подчинительный союз",
    "SYM": "Символ",
    "VERB": "Глагол",
    "X": "Другое",
}
PYMORPHY_POS_RU = {
    "NOUN": "Существительное",
    "ADJF": "Прилагательное",
    "ADJS": "Краткое прилагательное",
    "COMP": "Сравнительная степень",
    "VERB": "Глагол",
    "INFN": "Инфинитив",
    "PRTF": "Причастие",
    "PRTS": "Краткое причастие",
    "GRND": "Деепричастие",
    "NUMR": "Числительное",
    "ADVB": "Наречие",
    "NPRO": "Местоимение",
    "PRED": "Предикатив",
    "PREP": "Предлог",
    "CONJ": "Союз",
    "PRCL": "Частица",
    "INTJ": "Междометие",
}


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
        self.nlp = self._init_stanza()
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

        summary_frame = ttk.LabelFrame(main_frame, text="Сводка по частям речи")
        summary_frame.grid(row=4, column=0, columnspan=2, sticky="nsew", pady=(10, 0))
        summary_frame.columnconfigure(0, weight=1)

        self.total_words_label = ttk.Label(summary_frame, text="Всего слов: 0")
        self.total_words_label.grid(row=0, column=0, sticky="w", padx=5, pady=(5, 2))

        self.pos_tree = ttk.Treeview(
            summary_frame,
            columns=("pos", "count", "coef"),
            show="headings",
            height=6,
        )
        self.pos_tree.heading("pos", text="Часть речи")
        self.pos_tree.heading("count", text="Количество")
        self.pos_tree.heading("coef", text="Коэффициент")
        self.pos_tree.column("pos", width=200, anchor="w")
        self.pos_tree.column("count", width=120, anchor="center")
        self.pos_tree.column("coef", width=120, anchor="center")
        self.pos_tree.grid(row=1, column=0, sticky="nsew", padx=5, pady=(0, 5))

        pos_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=self.pos_tree.yview)
        self.pos_tree.configure(yscrollcommand=pos_scrollbar.set)
        pos_scrollbar.grid(row=1, column=1, sticky="ns", pady=(0, 5))

    def _init_stanza(self) -> stanza.Pipeline | None:
        try:
            safe_globals = [
                np.core.multiarray._reconstruct,
                np.core.multiarray.scalar,
                np.dtype,
                np.ndarray,
            ]
            if hasattr(torch.serialization, "add_safe_globals"):
                torch.serialization.add_safe_globals(safe_globals)
            stanza.download("ru", verbose=False)
            if hasattr(torch.serialization, "safe_globals"):
                with torch.serialization.safe_globals(safe_globals):
                    return stanza.Pipeline(
                        "ru",
                        processors="tokenize,pos,lemma",
                        use_gpu=False,
                        tokenize_no_ssplit=True,
                        verbose=False,
                    )
            return stanza.Pipeline(
                "ru",
                processors="tokenize,pos,lemma",
                use_gpu=False,
                tokenize_no_ssplit=True,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - UI fallback
            messagebox.showwarning(
                "Контекстный анализ недоступен",
                f"Не удалось загрузить модель Stanza. Будет использован базовый анализ.\n{exc}",
            )
            return None
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

        if self.nlp:
            doc = self.nlp(text)
            for sentence in doc.sentences:
                for word in sentence.words:
                    if not WORD_RE.fullmatch(word.text or ""):
                        continue
                    lemma = (word.lemma or word.text).lower()
                    pos_label = UD_POS_RU.get(word.upos or "", "Неизвестно")
                    form_key = word.text.lower()

                    lemma_counts[lemma] += 1
                    if form_key not in lemma_forms[lemma]:
                        lemma_forms[lemma][form_key] = pos_label
                    pos_counts[pos_label] += 1
                    total_words += 1
        else:
            for match in WORD_RE.finditer(text):
                token = match.group(0)
                parse = self.morph.parse(token)[0]
                lemma = parse.normal_form
                pos_label = PYMORPHY_POS_RU.get(parse.tag.POS, "Неизвестно")
                form_key = token.lower()

                lemma_counts[lemma] += 1
                if form_key not in lemma_forms[lemma]:
                    lemma_forms[lemma][form_key] = pos_label
                pos_counts[pos_label] += 1
                total_words += 1

        for lemma, count in sorted(lemma_counts.items()):
            forms = lemma_forms[lemma]
            lemma_data[lemma] = LemmaEntry(lemma=lemma, count=count, forms=forms)

        self.entries = list(lemma_data.values())
        self.pos_counts = dict(pos_counts)
        self.total_words = total_words
        self._populate_table()
        self._populate_pos_summary()

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

        figure, ax = plt.subplots(figsize=(7, 5))
        labels = list(coefficients.keys())
        values = [coefficients[label] for label in labels]

        if not values:
            messagebox.showinfo("Нет данных", "Недостаточно данных для диаграммы.")
            return

        colors = plt.cm.tab20.colors
        ax.pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90,
            colors=colors[: len(values)],
            textprops={"fontsize": 9},
        )
        ax.set_title("Частотные коэффициенты частей речи")
        ax.axis("equal")
        figure.tight_layout()

        canvas = FigureCanvasTkAgg(figure, master=window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _populate_pos_summary(self) -> None:
        for item in self.pos_tree.get_children():
            self.pos_tree.delete(item)

        self.total_words_label.config(text=f"Всего слов: {self.total_words}")
        if self.total_words == 0:
            return

        for pos_label, count in sorted(self.pos_counts.items()):
            coefficient = count / self.total_words
            self.pos_tree.insert(
                "",
                "end",
                values=(pos_label, count, f"{coefficient:.4f}"),
            )

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
