"""Métricas de avaliação offline, sem acesso ao estado privilegiado."""

from collections import Counter

from engine.serialize import parse_output


def evaluate_outputs(house, targets, predictions):
    """Avalia pares ordenados de rótulos e previsões."""

    targets, predictions = list(targets), list(predictions)

    if not targets or len(targets) != len(predictions):
        raise ValueError("targets e predictions devem ter o mesmo tamanho não nulo")

    exact = room = valid = 0
    non_idle_correct = non_idle_total = 0

    tp = Counter()
    fp = Counter()
    fn = Counter()
    classes = set()

    for target, prediction in zip(targets, predictions):
        expected = parse_output(house, target)
        expected_actions = set(expected["do"])

        classes.update(expected_actions)

        if expected_actions:
            non_idle_total += 1

        try:
            actual = parse_output(house, prediction)
        except (ValueError, TypeError):
            # Uma resposta inválida não recebe crédito.
            fn.update(expected_actions)
            continue

        valid += 1

        actual_actions = set(actual["do"])
        classes.update(actual_actions)

        correct = actual_actions == expected_actions

        exact += correct
        room += actual["next"] == expected["next"]

        if expected_actions:
            non_idle_correct += correct

        # Contabiliza os acertos e erros por classe de ação.
        tp.update(expected_actions & actual_actions)
        fp.update(actual_actions - expected_actions)
        fn.update(expected_actions - actual_actions)

    # Calcula o F1 individual de cada classe.
    f1_per_class = {}

    for action in sorted(classes):
        denominator = 2 * tp[action] + fp[action] + fn[action]

        f1_per_class[action] = 2 * tp[action] / denominator if denominator else 0.0

    macro_f1 = sum(f1_per_class.values()) / len(f1_per_class) if f1_per_class else None

    n = len(targets)

    return {
        "n": n,
        "exact_action_match": exact / n,
        "next_room_accuracy": room / n,
        "format_validity": valid / n,
        "non_idle_n": non_idle_total,
        "non_idle_exact_action_match": (
            non_idle_correct / non_idle_total if non_idle_total else None
        ),
        "macro_f1": macro_f1,
        "f1_per_class": f1_per_class,
    }
