import numpy as np
from perturbations import perturb_move
from perturbations.order_changes import move_cities


def ls_first_improvement(
    fitness_fn, initialisation_fn, perturbation_fn, max_steps=200, max_evals=5000
):
    """
    Local search alg. using a given fitness function, initialisation function, perturbation function and distance matrix.
    The algorithm will run for a maximum number of steps or evaluations, whichever comes first.
    The algorithm will return the best solution found so far, along with its fitness value and the history of all solutions evaluated.
    """

    # where we store partial results from each iter.
    iterated_solutions = []

    # initialisation
    current_solution = initialisation_fn()  # curried for the specific size beforehand!
    current_fitness = fitness_fn(current_solution)
    iteration = 0
    evals = 0

    overall_best = current_solution
    overall_best_fitness = current_fitness
    # try to move, based on some perturbation function. is this better? if so, move there, if not, try again
    while True:
        iteration += 1
        if iteration > max_steps or evals > max_evals:  # whichever comes first
            break

        local_evals = 0
        while True:
            evals += 1
            local_evals += 1
            if evals > max_evals:  # will break again in the higher loop
                break

            # perturb the current solution
            candidate_solution = perturbation_fn(current_solution)
            # calculate and compare fitness
            candidate_fitness = fitness_fn(candidate_solution)

            # update the overall best solution
            if candidate_fitness < overall_best_fitness:
                overall_best = candidate_solution
                overall_best_fitness = candidate_fitness

            # store and go to the next iteration, even when going to the worse
            if candidate_fitness < current_fitness or local_evals >= 1500:
                current_solution = candidate_solution
                current_fitness = candidate_fitness
                iterated_solutions.append(
                    {
                        "iteration": iteration,
                        "fitness": current_fitness,
                        "solution": current_solution,
                        "local_evals": local_evals,
                    }
                )
                break

    return {
        "history": iterated_solutions,
        "best_order": overall_best,
        "best_fitness": overall_best_fitness,
    }


def ls_best_improvement(
    fitness_fn,
    initialisation_fn,
    perturbation_fn,
    max_steps=200,
    max_evals=5000,  # not important here
):
    """
    Local search alg. using a given fitness function, initialisation function, perturbation function and distance matrix.
    The algorithm will run for a maximum number of steps or evaluations, whichever comes first.
    The algorithm will return the best solution found so far, along with its fitness value and the history of all solutions evaluated.
    """

    # where we store partial results from each iter.
    iterated_solutions = []

    # initialisation
    current_solution = initialisation_fn()  # curried for the specific size beforehand!
    current_fitness = fitness_fn(current_solution)
    iteration = 0

    overall_best = current_solution
    overall_best_fitness = current_fitness

    # try to move, based on some perturbation function. is this better? if so, move there, if not, try again
    while iteration < max_steps:
        iteration += 1

        all_perturbs = np.array(perturbation_fn(current_solution))
        all_fitness = np.array([fitness_fn(p) for p in all_perturbs])

        best_idx = np.argmin(all_fitness)
        best_fitness = all_fitness[best_idx]
        best_order = all_perturbs[best_idx]

        # store and go to the next iteration, even when going to the worse
        restarted = False
        if best_fitness < current_fitness:
            current_solution = best_order
            current_fitness = best_fitness

            # update the overall best solution
            if current_fitness < overall_best_fitness:
                overall_best = current_solution
                overall_best_fitness = current_fitness
        else:  # no betterm restart the problem
            restarted = True
            # current_solution = initialisation_fn()
            current_solution = all_perturbs[np.random.randint(len(all_perturbs))]
            current_fitness = fitness_fn(current_solution)

        iterated_solutions.append(
            {
                "iteration": iteration,
                "fitness": current_fitness,
                "solution": current_solution,
                "restart here": restarted,
            }
        )

    return {
        "history": iterated_solutions,
        "best_order": overall_best,
        "best_fitness": overall_best_fitness,
    }
