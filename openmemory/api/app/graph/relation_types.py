"""
Relationship Type Kategorien für OM_RELATION Kanten.

Diese Kategorien dienen als Query-Filter für graph_entity_relations().
Das LLM kann beliebige Beziehungstypen extrahieren - diese Liste ist
nicht exhaustiv und schränkt die Extraktion nicht ein.

Kategorien:
- family: Verwandtschaftsbeziehungen
- social: Freundschaften, Partnerschaften
- work: Berufliche Beziehungen
- location: Ortsbeziehungen
- creative: Projekte, Werke
- membership: Mitgliedschaften
- travel: Reisen, Besuche
"""

from typing import Dict, List


# Bekannte Kategorien für Query-Filter (nicht exhaustiv)
# Das LLM kann beliebige Types extrahieren - unbekannte fallen in keine Kategorie
RELATION_CATEGORIES: Dict[str, List[str]] = {
    "family": [
        "eltern_von",
        "kind_von",
        "mutter_von",
        "vater_von",
        "schwester_von",
        "bruder_von",
        "verwandt_mit",
        "onkel_von",
        "tante_von",
        "cousin_von",
        "grosseltern_von",
        "enkel_von",
    ],
    "social": [
        "partner_von",
        "freund_von",
        "mitbewohner_von",
        "bekannt_mit",
        "verlobt_mit",
        "verheiratet_mit",
    ],
    "work": [
        "arbeitet_bei",
        "arbeitete_bei",
        "kollege_von",
        "arbeitspartner_von",
        "gruendete",
        "leitet",
        "angestellt_bei",
        "chef_von",
        "mitarbeiter_von",
    ],
    "location": [
        "wohnt_in",
        "wohnte_in",
        "geboren_in",
        "aufgewachsen_in",
        "befindet_sich_in",
        "lebt_in",
        "stammt_aus",
    ],
    "creative": [
        "produzierte",
        "wirkte_mit_in",
        "regie_bei",
        "schrieb",
        "erschuf",
        "komponierte",
        "entwickelte",
    ],
    "membership": [
        "mitglied_von",
        "aktiv_in",
        "engagiert_in",
        "teil_von",
        "gehoert_zu",
    ],
    "travel": [
        "plant_besuch",
        "reist_nach",
        "besucht",
        "fliegt_nach",
        "faehrt_nach",
        "war_in",
    ],
}


# Inverse Beziehungen für bidirektionale Traversierung
# Symmetrische Beziehungen verweisen auf sich selbst
INVERSE_RELATIONS: Dict[str, str] = {
    # Family
    "eltern_von": "kind_von",
    "kind_von": "eltern_von",
    "mutter_von": "kind_von",
    "vater_von": "kind_von",
    "schwester_von": "schwester_von",  # symmetric
    "bruder_von": "bruder_von",        # symmetric
    "verwandt_mit": "verwandt_mit",    # symmetric
    "onkel_von": "neffe_von",
    "tante_von": "neffe_von",
    "grosseltern_von": "enkel_von",
    "enkel_von": "grosseltern_von",
    # Social
    "partner_von": "partner_von",      # symmetric
    "freund_von": "freund_von",        # symmetric
    "mitbewohner_von": "mitbewohner_von",  # symmetric
    "bekannt_mit": "bekannt_mit",      # symmetric
    "verheiratet_mit": "verheiratet_mit",  # symmetric
    "verlobt_mit": "verlobt_mit",      # symmetric
    # Work
    "kollege_von": "kollege_von",      # symmetric
    "chef_von": "mitarbeiter_von",
    "mitarbeiter_von": "chef_von",
}


# Mapping von englischen/alternativen Schreibweisen zu deutschen Standard-Types
TYPE_ALIASES: Dict[str, str] = {
    # English to German
    "is_mother_of": "mutter_von",
    "is_father_of": "vater_von",
    "is_child_of": "kind_von",
    "is_sibling_of": "verwandt_mit",
    "is_sister_of": "schwester_von",
    "is_brother_of": "bruder_von",
    "works_at": "arbeitet_bei",
    "worked_at": "arbeitete_bei",
    "lives_in": "wohnt_in",
    "lived_in": "wohnte_in",
    "born_in": "geboren_in",
    "member_of": "mitglied_von",
    "married_to": "verheiratet_mit",
    # Short forms
    "partner": "partner_von",
    "friend": "freund_von",
    "roommate": "mitbewohner_von",
    "colleague": "kollege_von",
    "mother": "mutter_von",
    "father": "vater_von",
    "sister": "schwester_von",
    "brother": "bruder_von",
}


def normalize_relation_type(raw_type: str) -> str:
    """
    Normalisiert einen Relation-Type zu einer Standard-Form.

    Unbekannte Types werden durchgelassen (lowercase, underscores).

    Examples:
        "is_mother_of" → "mutter_von"
        "works at" → "arbeitet_bei"
        "PARTNER" → "partner_von"
        "some_new_relation" → "some_new_relation"  # unbekannt, aber OK
    """
    normalized = raw_type.lower().strip()
    normalized = normalized.replace(" ", "_")
    normalized = normalized.replace("-", "_")

    # Bekanntes Alias? Dann mappen, sonst durchlassen
    return TYPE_ALIASES.get(normalized, normalized)


def get_relations_for_category(category: str) -> List[str]:
    """
    Gibt alle bekannten Relation-Types für eine Kategorie zurück.

    Returns:
        Liste von Relation-Types, oder leere Liste wenn Kategorie unbekannt
    """
    return RELATION_CATEGORIES.get(category.lower(), [])


def get_category_for_relation(relation_type: str) -> str | None:
    """
    Gibt die Kategorie für einen Relation-Type zurück.

    Returns:
        Kategorie-Name oder None wenn Type in keiner Kategorie
    """
    for category, relations in RELATION_CATEGORIES.items():
        if relation_type in relations:
            return category
    return None


def get_inverse_relation(relation_type: str) -> str | None:
    """
    Gibt die inverse Beziehung zurück.

    Returns:
        Inverse Relation oder None wenn keine definiert
    """
    return INVERSE_RELATIONS.get(relation_type)


def is_symmetric_relation(relation_type: str) -> bool:
    """
    Prüft ob eine Beziehung symmetrisch ist (A→B impliziert B→A).

    Unbekannte Relations werden als nicht-symmetrisch angenommen.
    """
    inverse = INVERSE_RELATIONS.get(relation_type)
    return inverse == relation_type


def get_all_known_types() -> List[str]:
    """Gibt alle bekannten (kategorisierten) Relation-Types zurück."""
    all_types = []
    for types in RELATION_CATEGORIES.values():
        all_types.extend(types)
    return list(set(all_types))


def get_all_categories() -> List[str]:
    """Gibt alle Kategorien zurück."""
    return list(RELATION_CATEGORIES.keys())
