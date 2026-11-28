"""Métricas de avaliação offline, sem acesso ao estado privilegiado."""
from engine.serialize import parse_output


def evaluate_outputs(house, targets, predictions):
    """Avalia pares ordenados de rótulos e previsões; saída inválida conta como erro."""
    targets, predictions = list(targets), list(predictions)
    if not targets or len(targets) != len(predictions):
        raise ValueError("targets e predictions devem ter o mesmo tamanho não nulo")
    exact = room = valid = non_idle_correct = non_idle_total = 0
    for target, prediction in zip(targets, predictions):
        expected = parse_output(house, target)
        if expected['do']:
            non_idle_total += 1
        try:
            actual = parse_output(house, prediction)
        except (ValueError, TypeError):
            continue
        valid += 1
        exact += actual['do'] == expected['do']
        room += actual['next'] == expected['next']
        if expected['do']:
            non_idle_correct += actual['do'] == expected['do']
    n = len(targets)
    return {
        'n': n,
        'exact_action_match': exact / n,
        'next_room_accuracy': room / n,
        'format_validity': valid / n,
        'non_idle_n': non_idle_total,
        'non_idle_exact_action_match': non_idle_correct / non_idle_total if non_idle_total else None,
    }
