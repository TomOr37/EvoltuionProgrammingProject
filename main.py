from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from birdServices import BirdCreator, BirdEvaluator
from operators import ModelParamSwapCrossOver, ModelAddDistMutation


def main():
    algo = SimpleEvolution(
        Subpopulation(creators=BirdCreator(init_pos=(230, 350)),
                      population_size=40,
                      # user-defined fitness evaluation method
                      evaluator=BirdEvaluator(),
                      # maximization problem (fitness is sum of values), so higher fitness is better
                      higher_is_better=True,
                      elitism_rate=1 / 300,
                      # genetic operators sequence to be applied in each generation
                      operators_sequence=[
                          ModelParamSwapCrossOver(probability=0.1),
                          ModelAddDistMutation(probability=0.1)
                      ],
                      selection_methods=[
                          # (selection method, selection probability) tuple
                          (TournamentSelection(tournament_size=3, higher_is_better=True), 1)
                      ]
                      ),
        breeder=SimpleBreeder(),
        max_workers=5,
        max_generation=2,

        termination_checker=ThresholdFromTargetTerminationChecker(optimal=500, threshold=0.0),
        statistics=BestAverageWorstStatistics()
    )

    # evolve the generated initial population
    algo.evolve()

    # Execute (show) the best solution
    print(algo.execute())


if __name__ == '__main__':
    main()
