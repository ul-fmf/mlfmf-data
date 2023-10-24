from typing import List, Optional
import re
from apaa.data.structures.agda_fact import AgdaFact
from apaa.other.helpers import Other, AgdaSyntax, TextManipulation


LOGGER = Other.create_logger(__file__)


class AgdaFileParser:
    @staticmethod
    def parse_all_files(root_dir: str) -> List[AgdaFact]:
        facts = []
        for file in Other.get_all_files(root_dir):
            facts.extend(AgdaFileParser.parse_file(file))
        return facts

    @staticmethod
    def parse_file(file: str) -> List[AgdaFact]:
        """
        Extracts interesting parts (definitions) and parses them by calling
        parse_file_part.

        An interesting part is something like

        empty line
        name : type
            optional lines
        definition (contains name)
            optional lines

        where name is not among the keywords (skip types and stuff like that)

        :param file:
        :return:
        """
        with open(file, encoding="utf-8") as f:
            lines = f.readlines()
        lines = AgdaFileParser.remove_comments(lines)
        return AgdaFileParser.parse_file_lines(AgdaFileParser._file_to_module(file), lines)

    @staticmethod
    def _file_to_module(file: str):
        agda_extension = ".agda"
        if not file.endswith(agda_extension):
            raise ValueError(f"File {file} should end with {agda_extension}")
        return re.sub(r"[/\\]+", ".", file[:-len(agda_extension)])

    @staticmethod
    def parse_file_lines(file, lines: List[str]) -> List[AgdaFact]:
        parts: List[AgdaFact] = []
        current_part: List[str] = []
        last_indent = 0
        last_line_empty = False
        for line in lines:
            this_indent = AgdaFileParser._compute_indent(line)
            line = line.rstrip()
            word = AgdaFileParser.first_word(line)
            if this_indent == 0:
                # unless ^, a new part cannot begin.
                finished_an_indented_block = last_indent > 0 and last_line_empty and line
                finished_non_intended_block = last_indent == 0 and last_line_empty and line
                start_of_a_new_block = word in AgdaSyntax.KEYWORDS
                process_previous = finished_an_indented_block or \
                    finished_non_intended_block or \
                    start_of_a_new_block
                must_continue = word in AgdaSyntax.BLOCK_CONTINUATION
                if current_part and process_previous and not must_continue:
                    # an indented block ended in the previous line
                    parts.extend(AgdaFileParser.parse_file_part(file, current_part))
                    current_part = []
            if line:
                if this_indent > 0 or \
                        word not in AgdaSyntax.KEYWORDS or \
                        word in AgdaSyntax.BLOCK_STARTERS:
                    current_part.append(line)
                last_indent = this_indent
            else:
                if current_part:
                    current_part.append(line)  # need empty lines due to recursive calls
            last_line_empty = not line
        if current_part:
            parts.extend(AgdaFileParser.parse_file_part(file, current_part))
        parts = list(filter(lambda p: p is not None, parts))
        return parts

    @staticmethod
    def _compute_indent(line: str):
        indent = 0
        for c in line:
            if c == " ":
                indent += 1
            elif c == "\t":
                indent += 4
            else:
                break
        return indent

    @staticmethod
    def first_word(string):
        string = string.strip()
        i = string.find(" ")
        return string if i < 0 else string[:i]

    @staticmethod
    def remove_comments(lines: List[str]) -> List[str]:
        """
        Removes single-line comments and possibly nested multi-line comments.
        Removes also newline character. Removes more than one consecutive empty line (after
        removing the comments).

        :param lines:

        :return:
        """
        multi_line_comment_depth = 0
        string_stack = []
        better_lines = []
        last_line_not_empty = False
        escaping = False
        for line in lines:
            better_line = []
            i = 0
            while i < len(line):
                this = line[i]
                this_and_next = line[i: i + 2]
                if this_and_next == "{-" and not string_stack:
                    multi_line_comment_depth += 1
                    i += 2
                elif this_and_next == "-}" and not string_stack:
                    multi_line_comment_depth -= 1
                    i += 2
                elif this_and_next == "--" and not (string_stack or multi_line_comment_depth):
                    i = len(line)
                else:
                    AgdaFileParser._update_string_stack(line, this, i, escaping, string_stack)
                    escaping = this == "\\" and not escaping
                    if not multi_line_comment_depth and this != "\n":
                        better_line.append(this)
                    i += 1
            better_line_joined = "".join(
                better_line).rstrip()  # to prevent adding what is left of ^  -- comment
            if better_line_joined or (not line.strip() and last_line_not_empty):
                better_lines.append(better_line_joined)
                last_line_not_empty = bool(better_line_joined)
        if string_stack or escaping:
            raise ValueError(
                f"String stack {string_stack} and escaping {escaping} should evaluate to False.")
        return better_lines

    @staticmethod
    def _update_string_stack(line: str, this: str, i: int, escaping: bool, string_stack: List[str]):
        if this in ['"', "'"] and not escaping:
            if string_stack and this == string_stack[-1]:
                string_stack.pop()
            elif not string_stack:
                if this == "'" and line[:i].endswith("char ") or this == '"':
                    string_stack.append(this)

    @staticmethod
    def parse_file_part(file: str, lines: List[str]) -> List[Optional[AgdaFact]]:
        first_word = AgdaFileParser.first_word(lines[0])
        if first_word in AgdaSyntax.BLOCK_STARTERS:
            if first_word not in AgdaSyntax.USED_BLOCK_STARTERS:
                return [None]
            elif first_word == "record":
                declaration_lines, description_lines = AgdaFileParser.split_until_where(lines)
                if not (declaration_lines and description_lines):
                    return [None]
                return [AgdaFact(file, declaration_lines, description_lines)]
            elif first_word in ["module", "macro", "instance"]:
                # parse rest of the part with decreased indent
                if first_word == "module":
                    the_word = "where"
                    file += "." + AgdaFileParser._extract_module_name(" ".join(lines))
                else:
                    the_word = first_word
                lines_decreased = AgdaFileParser._decrease_indent(lines, the_word)
                return AgdaFileParser.parse_file_lines(file, lines_decreased)
            else:
                raise ValueError(f"Unexpected first word: {first_word} for {lines}")
        elif all(":" not in line for line in lines):
            if sum(bool(line) for line in lines) > 1:
                LOGGER.warning(f"Problem in {file}: {lines}. Defining unknown ...")
            return [AgdaFact(file, [AgdaFact.UNKNOWN], lines)]

        declaration_lines = []
        description_lines = []
        current_part = declaration_lines
        for line in lines:
            if current_part and not line.startswith(" ") and current_part is declaration_lines:
                current_part = description_lines
            current_part.append(line)
        return [AgdaFact(file, declaration_lines, description_lines)]

    @staticmethod
    def _extract_module_name(line: str):
        return re.search("module +([^ ]+) ?", line).group(1)

    @staticmethod
    def _decrease_indent(lines: List[str], word: str) -> List[str]:
        """
        Keeps only the lines after the first occurrence of the word.
        The indent of those is defined so that the minimal indent becomes zero.

        :param lines:
        :param word:
        :return:
        """

        i_start = -1
        for i, line in enumerate(lines):
            if word in line:
                i_start = i + 1
                break
        if i_start < 0:
            LOGGER.warning(f"where was not found in {lines}")
            i_start = len(lines)
        while i_start < len(lines) and not lines[i_start]:
            i_start += 1
        filtered_lines = lines[i_start:]
        min_indent = 10 ** 15  # a very long line :)
        for line in filtered_lines:
            if line:
                min_indent = min(AgdaFileParser._compute_indent(line), min_indent)
        for i, line in enumerate(filtered_lines):
            filtered_lines[i] = line[min_indent:]
        if lines == filtered_lines:
            _ = 21
        return filtered_lines

    @staticmethod
    def split_until_where(lines: List[str]):
        i_end = 0
        while i_end < len(lines) and "where" not in lines[i_end]:
            i_end += 1
        return lines[:i_end + 1], lines[i_end + 1:]


def test_comment_removal():
    with open("test_data/comments.agda", encoding="utf-8") as f:
        lines = f.readlines()
    for line in AgdaFileParser.remove_comments(lines):
        LOGGER.info(line)


def test_parse_one():
    AgdaFileParser.parse_file("test_data/parts.agda")
    AgdaFileParser.parse_file(r"../agda-stdlib/src\Algebra\Apartness\Bundles.agda")
    AgdaFileParser.parse_file(r"../agda-stdlib/src\Text\Pretty.agda")
    AgdaFileParser.parse_file("../agda-stdlib/src\\Codata\\Sized\\Delay\\Properties.agda")
    AgdaFileParser.parse_file(r"../agda-stdlib/src\Text\Regex\Search.agda")


def test_name_split():
    name = "this₂₂Is_aVery-longName_innit12∘21"
    expected = ['this', '₂₂', 'Is', 'a', 'Very', 'long', 'Name', 'innit', '12', '∘', '21']
    assert expected == TextManipulation.name_to_parts(name)


if __name__ == "__main__":
    test_parse_one()
