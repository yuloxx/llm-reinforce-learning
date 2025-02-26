from typing import Any, List, Tuple, Optional


def select_node_by_id(node_id: int, g: Any) -> Optional[dict]:
    """
    Selects a node from the graph by its unique ID.

    Args:
        node_id (int): The ID of the node to find.
        g (Any): The graph data structure containing nodes.

    Returns:
        Optional[dict]: The node dictionary if found, otherwise None.
    """
    node_list = [node for node in g['nodes'] if node['id'] == node_id]
    return node_list[0] if node_list else None


def select_node_by_classname(class_name: str, g: Any) -> Optional[dict]:
    """
    Selects the first node with the given class name.

    Args:
        class_name (str): The class name of the node to find.
        g (Any): The graph data structure containing nodes.

    Returns:
        Optional[dict]: The first node dictionary that matches the class name, or None if not found.
    """
    node_list = [node for node in g['nodes'] if node['class_name'] == class_name]
    return node_list[0] if node_list else None


def query_node_classname_by_id(node_id: int, g: Any) -> Optional[str]:
    """
    Retrieves the class name of a node given its ID.

    Args:
        node_id (int): The ID of the node.
        g (Any): The graph data structure.

    Returns:
        Optional[str]: The class name of the node if found, otherwise None.
    """
    node = select_node_by_id(node_id, g)
    return node['class_name'] if node else None


def query_node_id_by_classname(class_name: str, g: Any) -> Optional[int]:
    """
    Retrieves the ID of a node given its class name.

    Args:
        class_name (str): The class name of the node.
        g (Any): The graph data structure.

    Returns:
        Optional[int]: The ID of the node if found, otherwise None.
    """
    node = select_node_by_classname(class_name, g)
    return node['id'] if node else None


def select_relations_by_node_id(node_id: int, g: Any) -> Tuple[List[dict], List[dict]]:
    """
    Selects all relations (edges) connected to a node by its ID.

    Args:
        node_id (int): The ID of the node.
        g (Any): The graph data structure containing edges.

    Returns:
        Tuple[List[dict], List[dict]]: A tuple containing:
            - List of outgoing relations (edges where node_id is from_id).
            - List of incoming relations (edges where node_id is to_id).
    """
    from_rels = [edge for edge in g['edges'] if edge['from_id'] == node_id]
    to_rels = [edge for edge in g['edges'] if edge['to_id'] == node_id]
    return from_rels, to_rels


def format_from_relation(node_id: int, from_relation: List[Any], g: Any) -> List[str]:
    """
    Formats outgoing relations from a given node ID.

    Args:
        node_id (int): The ID of the node.
        from_relation (List[Any]): List of outgoing relations.
        g (Any): The graph data structure.

    Returns:
        List[str]: A list of formatted relation strings.
    """
    node_class_name = query_node_classname_by_id(node_id, g)
    return [
        f'from: {node_class_name} rels: {edge["relation_type"]} to: {query_node_classname_by_id(edge["to_id"], g)}'
        for edge in from_relation
    ]


def format_to_relation(node_id: int, to_relation: List[Any], g: Any) -> List[str]:
    """
    Formats incoming relations to a given node ID.

    Args:
        node_id (int): The ID of the node.
        to_relation (List[Any]): List of incoming relations.
        g (Any): The graph data structure.

    Returns:
        List[str]: A list of formatted relation strings.
    """
    node_class_name = query_node_classname_by_id(node_id, g)
    return [
        f'from: {query_node_classname_by_id(edge["from_id"], g)} rels: {edge["relation_type"]} to: {node_class_name}'
        for edge in to_relation
    ]


def query_relations_by_node_id(node_id: int, g: Any) -> Tuple[List[str], List[str]]:
    """
    Queries and formats both incoming and outgoing relations for a node.

    Args:
        node_id (int): The ID of the node.
        g (Any): The graph data structure.

    Returns:
        Tuple[List[str], List[str]]: Formatted outgoing and incoming relation strings.
    """
    from_rels, to_rels = select_relations_by_node_id(node_id, g)
    from_rels_str = format_from_relation(node_id, from_rels, g)
    to_rels_str = format_to_relation(node_id, to_rels, g)
    return from_rels_str, to_rels_str


def select_character(g: Any) -> List[dict]:
    """
    Selects all nodes that represent characters in the graph.

    Args:
        g (Any): The graph data structure.

    Returns:
        List[dict]: A list of character nodes.
    """
    return [node for node in g['nodes'] if node['class_name'] == 'character']


def query_character_id(g: Any) -> List[int]:
    """
    Retrieves all character node IDs.

    Args:
        g (Any): The graph data structure.

    Returns:
        List[int]: A list of character node IDs.
    """
    character_list = select_character(g)
    return [character['id'] for character in character_list]


def select_character_relations(character_id: int, g: Any) -> List[dict]:
    """
    Selects relations where the character node is the source.

    Args:
        character_id (int): The ID of the character node.
        g (Any): The graph data structure.

    Returns:
        List[dict]: A list of relations where the character is the source.
    """
    rels, _ = select_relations_by_node_id(character_id, g)
    return rels


def query_character_relations(character_id: int, g: Any) -> List[str]:
    """
    Queries and formats relations where the character node is the source.

    Args:
        character_id (int): The ID of the character node.
        g (Any): The graph data structure.

    Returns:
        List[str]: Formatted relations where the character is the source.
    """
    rels_str, _ = query_relations_by_node_id(character_id, g)
    return rels_str
