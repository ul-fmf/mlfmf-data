import re
from collections import Counter
from typing import List, Optional, Union

from .agda_tree import AgdaDefinition
from .knowledge_graph import KnowledgeGraph
from apaa.other.helpers import MyTypes, NodeType, TextManipulation


class AgdaFact:
    """
    This is more or less deprecated - used only in simple parser
    that was created before the real one.
    """
    UNKNOWN = "unknown"
    TEMP = "temp"
    N_UNKNOWN = 0

    def __init__(
        self,
        file: str,
        declaration: Optional[List[str]] = None,
        body: Optional[List[str]] = None,
        tree: Optional[AgdaDefinition] = None,
        name: Optional[MyTypes.NODE] = None,
        kg: Optional[KnowledgeGraph] = None,
    ):
        self._file = file
        self._lines: List[List[str]] = []
        self._text: List[List[str]] = []
        self._name: Union[MyTypes.NODE, None] = name  # same as in graph
        self._words_declaration: Counter[str] = Counter()
        self._words_body: Counter[str] = Counter()
        self._words: Counter[str] = Counter()
        self._kind: Optional[NodeType] = None

        can_use_tree = tree is not None and kg is not None
        can_use_text = declaration is not None and body is not None
        if can_use_tree:
            self._init_from_tree(tree, kg)
        elif can_use_text:
            self._init_from_text(declaration, body)
        else:
            raise ValueError(
                "Either (tree and kg) or (declaration and body) should not be None."
            )

    def _init_from_tree(self, definition: AgdaDefinition, kg: KnowledgeGraph):
        relevant_nodes = []
        if not AgdaDefinition.is_with_definition(definition.name[2]):
            relevant_nodes.append(definition.name)
            relevant_nodes.extend(kg.get_helper_nodes(definition.name))
        declaration_parts = []
        body_parts = []
        for node in relevant_nodes:
            related_definition = kg.id_to_definition[node]
            declaration_parts.append(related_definition.type.full_text)
            body_parts.append(related_definition.body.full_text)
        declaration = AgdaFact._normalize_tree_text(" ".join(declaration_parts))
        body = AgdaFact._normalize_tree_text(" ".join(body_parts))
        self.name = definition.name
        self.kind = definition.body.node_type
        self._init_from_text([declaration], [body])

    @staticmethod
    def _normalize_tree_text(text: str):
        text = re.sub("[()]", " ", text)
        text = re.sub(" +", " ", text)
        return text.strip()

    def _init_from_text(self, declaration, body):
        self._lines = [declaration, body]
        self._text = [
            TextManipulation.extract_words(declaration),
            TextManipulation.extract_words(body),
        ]
        self._words_declaration = Counter(self._text[0])
        self._words_body = Counter(self._text[1])
        self._words = self._words_declaration
        self._words.update(self._words_body)

    @property
    def file(self):
        return self._file

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name: Optional[MyTypes.NODE]):
        if self._name is None:
            self._name = name

    @property
    def kind(self):
        return self._kind

    @kind.setter
    def kind(self, kind: NodeType):
        if self._kind is None:
            if kind.is_definition_type():
                self._kind = kind
            else:
                raise ValueError(
                    f"Definition {self.name} is not of definition type: {kind}"
                )

    @property
    def fully_qualified_name(self):
        return f"{self.file}.{self.name}"

    @property
    def words_declaration(self) -> Counter:
        return self._words_declaration

    @property
    def words_body(self) -> Counter:
        return self._words_body

    @property
    def words(self) -> Counter:
        return self._words

    @property
    def text_declaration(self) -> List[str]:
        return self._text[0]

    @property
    def text_body(self) -> List[str]:
        return self._text[1]

    @property
    def text(self):
        return self.text_declaration + self.text_body

    @property
    def lines(self):
        return self._lines

    def __repr__(self):
        return f"AgdaFact({self.fully_qualified_name}, {self.words})"

    def __str__(self):
        return "\n----declaration-body-separator----\n".join(
            ["\n".join(part) for part in self._lines]
        )
