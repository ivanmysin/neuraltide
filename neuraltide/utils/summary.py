import tensorflow as tf
from typing import Optional


def print_summary(network, rich_library: Optional[object] = None) -> None:
    """
    Выводит summary модели.

    Если библиотека rich доступна, использует её для форматирования,
    иначе fallback на ASCII.
    """
    try:
        from rich.console import Console
        from rich.table import Table
        has_rich = True
    except ImportError:
        has_rich = False

    if has_rich:
        _print_summary_rich(network)
    else:
        _print_summary_ascii(network)


def _print_summary_ascii(network) -> None:
    """ASCII fallback для print_summary."""
    print("=" * 80)
    print("NEURALTIDE MODEL SUMMARY")
    print("=" * 80)

    print("\nINPUT POPULATIONS")
    print("-" * 80)
    print(f"{'Name':<15} {'Generator':<25} {'n_units':<10}")
    print("-" * 80)

    graph = network._graph
    for name in graph.input_population_names:
        pop = graph._populations[name]
        gen_name = pop.generator.name
        print(f"{name:<15} {gen_name:<25} {pop.n_units:<10}")

    print("\nDYNAMIC POPULATIONS")
    print("-" * 80)
    print(f"{'Name':<15} {'Model':<25} {'n_units':<10}")
    print("-" * 80)

    for name in graph.dynamic_population_names:
        pop = graph._populations[name]
        model_name = pop.__class__.__name__
        print(f"{name:<15} {model_name:<25} {pop.n_units:<10}")

    print("\nPROJECTIONS")
    print("-" * 80)
    print(f"{'Name':<20} {'src → tgt':<25} {'Shape':<15} {'Synapse':<15}")
    print("-" * 80)

    for syn_name, entry in graph._synapses.items():
        shape = f"[{entry.model.n_pre}, {entry.model.n_post}]"
        syn_type = entry.model.__class__.__name__
        connection = f"{entry.src} → {entry.tgt}"
        print(f"{syn_name:<20} {connection:<25} {shape:<15} {syn_type:<15}")

    trainable_count = sum(
        1 for var in network.trainable_variables if var.trainable
    )
    non_trainable_count = sum(
        1 for var in network.trainable_variables if not var.trainable
    )

    print("\n" + "-" * 80)
    print(f"Trainable params: {trainable_count} | Non-trainable: {non_trainable_count}")
    print("=" * 80)


def _print_summary_rich(network) -> None:
    """Rich-версия print_summary."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    graph = network._graph

    table = Table(title="NEURALTIDE MODEL SUMMARY")

    table.add_column("Section", style="bold")
    table.add_column("Details")

    console.print(table)

    console.print("[bold]INPUT POPULATIONS[/bold]")
    for name in graph.input_population_names:
        pop = graph._populations[name]
        console.print(f"  {name}: {pop.generator.name} (n_units={pop.n_units})")

    console.print("[bold]DYNAMIC POPULATIONS[/bold]")
    for name in graph.dynamic_population_names:
        pop = graph._populations[name]
        console.print(f"  {name}: {pop.__class__.__name__} (n_units={pop.n_units})")

    console.print("[bold]PROJECTIONS[/bold]")
    for syn_name, entry in graph._synapses.items():
        connection = f"{entry.src} → {entry.tgt}"
        shape = f"[{entry.model.n_pre}, {entry.model.n_post}]"
        console.print(f"  {syn_name}: {connection} {shape} ({entry.model.__class__.__name__})")

    trainable_count = sum(1 for var in network.trainable_variables if var.trainable)
    console.print(f"\n[bold]Trainable params:[/bold] {trainable_count}")
