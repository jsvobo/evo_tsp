import numpy as np
from representation import order_is_valid


def tournament_selection(population, num_children, fitness_list, tournament_size):
    """
    Select parents and produce children
    :param population: list of individuals
    :param num_parents: number of parents to select
    :param num_children: number of children to produce
    :param fitness_fn: fitness function
    :return: list of children
    """
    population_size = len(population)
    bag_size = min(tournament_size, population_size)
    selected = []
    for i in range(num_children):
        # sample random indices 0,len(population)
        idcs = np.random.randint(0, population_size, bag_size)
        best_idx = np.argmin(fitness_list[idcs])
        selected.append(population[idcs[best_idx]])
    return selected


def combined_replacement(old, new, num_survivors, fitness_list_old, fitness_list_new):
    """
    Replace old population with new population
    :param old: list of individuals
    :param new: list of individuals
    :param num_survivors: number of survivors
    :param fitness_fn: fitness function
    :return: list of individuals
    """
    population_size = len(old)
    # combine old and new populations
    population = old + new
    fitness_list = np.concatenate((fitness_list_old, fitness_list_new))

    # sort population by fitness
    idcs = np.argsort(fitness_list)
    # select survivors as ones with the best fitness
    smaller_list = idcs[:num_survivors]
    return [population[i] for i in smaller_list], fitness_list[smaller_list]


def initialisation(init_fn, generation_size):
    """
    Create initial population
    :param init_fn: function to create an individual
    :param generation_size: number of individuals in the population
    :return: list of individuals
    """
    return [init_fn() for _ in range(generation_size)]


def apply_crossover(distance_matrix, parent_list, operation, crossover_p):
    """
    Apply crossover to parents
    :param parents: list of individuals
    :param num_children: number of children to produce
    :return: list of children
    """
    children = []
    for i in range(len(parent_list) // 2):
        parent_a = parent_list[2 * i]
        parent_b = parent_list[2 * i + 1]
        if np.random.rand() < crossover_p:
            child = operation(distance_matrix, parent_a, parent_b)
            children.append(child)
    return children


def ea_alg(
    fitness_fn,  # have
    init_fn,  # have
    selection_fn,
    crossover_fn,
    mutation_fn,
    replacement_fn,
    generation_size,
    offspring_num,
    distance_matrix,
    tournament_size=10,
    p_cross=0.1,
    p_mut=0.05,
    max_evaluations=1000,
):

    results = []
    evaluations = generation_size

    # create initial population
    population = initialisation(init_fn, generation_size)
    fitness_list = np.array([fitness_fn(ind) for ind in population])
    best_idx = np.argmax(fitness_list)
    best_individual = population[best_idx]
    print("starting evolution:  best fitness of init", fitness_list[best_idx])
    generation = 0
    while evaluations <= max_evaluations:
        # select parents
        parents = selection_fn(
            population, offspring_num * 2, fitness_list, tournament_size
        )

        # crossover parents (right next to each other in the list)
        children = apply_crossover(distance_matrix, parents, crossover_fn, p_cross)
        evaluations += len(children)

        # mutate children
        children = [
            mutation_fn(child) if np.random.rand() < p_mut else child
            for child in children
        ]

        # evaluate children, replace population
        fitness_list_children = np.array([fitness_fn(ind) for ind in children])
        population, fitness_list = replacement_fn(
            population, children, generation_size, fitness_list, fitness_list_children
        )

        best_idx = np.argmin(fitness_list)
        if fitness_list[best_idx] < fitness_fn(best_individual):
            best_individual = population[best_idx]

        # cleanup, book-keeping
        results.append(
            {
                "generation": generation,
                "population": population,
                "avg_children_fitness": np.mean(fitness_list_children),
                "avg_fitness": np.mean(fitness_list),
                "best_individual": best_individual,
                "best_fitness": fitness_fn(best_individual),
                "evals": evaluations,
            }
        )
        print(
            f"Generation {generation}: {fitness_list[best_idx]}, {np.mean(fitness_list)}"
        )

        generation += 1

    return {
        "history": results,
        "best_order": best_individual,
        "best_fitness": fitness_fn(best_individual),
    }
