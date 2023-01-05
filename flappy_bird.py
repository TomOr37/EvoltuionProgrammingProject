"""
The classic game of flappy bird. Make with python
and pygame. Features pixel perfect collision using masks :o

"""
import pygame
import random
import os
import time
from eckity.algorithms.simple_evolution import SimpleEvolution
from eckity.breeders.simple_breeder import SimpleBreeder
from eckity.genetic_operators.selections.tournament_selection import TournamentSelection
from eckity.statistics.best_average_worst_statistics import BestAverageWorstStatistics
from eckity.subpopulation import Subpopulation
from eckity.termination_checkers.threshold_from_target_termination_checker import ThresholdFromTargetTerminationChecker
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.creators.creator import Creator
from eckity.fitness.simple_fitness import SimpleFitness
from feedForwardModel import FFModel
from eckity.individual import Individual
from operators import ModelAddDistMutation, ModelParamSwapCrossOver

pygame.font.init()  # init font

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = False

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs", "bg.png")).convert_alpha(), (600, 900))
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird" + str(x) + ".png"))) for x in
               range(1, 4)]
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")).convert_alpha())

IMGS = bird_images
"""
Note: the images should be in global scope, because in evaluation of the GP, serialization is made to individual,
 and pygame.Surface(The type of bird_images) is non-serializable
"""


class Bird(Individual):
    """
    Bird class representing the flappy bird
    """
    MAX_ROTATION = 25
    ROT_VEL = 20
    ANIMATION_TIME = 5

    def __init__(self, x, y, model, fitness):
        """
        Initialize the object
        :param x: starting x pos (int)
        :param y: starting y pos (int)
        :param model : random feed-forward net for determinate birds actions
        :param fitness : required param for genetic
        :return: None
        """
        super().__init__(fitness)
        self.x = x
        self.y = y
        self.tilt = 0  # degrees to tilt
        self.tick_count = 0
        self.vel = 0
        self.height = self.y
        self.img_count = 0
        self.model = model

    def jump(self):
        """
        make the bird jump
        :return: None
        """
        self.vel = -10.5
        self.tick_count = 0
        self.height = self.y

    def move(self):
        """
        make the bird move
        :return: None
        """
        self.tick_count += 1

        # for downward acceleration
        displacement = self.vel * (self.tick_count) + 0.5 * (3) * (self.tick_count) ** 2  # calculate displacement

        # terminal velocity
        if displacement >= 16:
            displacement = (displacement / abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:  # tilt up
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION
        else:  # tilt down
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    def draw(self, win):
        """
        draw the bird
        :param win: pygame window or surface
        :return: None
        """
        image_needed = getImage(self)

        # tilt the bird
        blitRotateCenter(win, image_needed, (self.x, self.y), self.tilt)

    def get_mask(self):
        """
        gets the mask for the current image of the bird
        :return: None
        """
        return pygame.mask.from_surface(getImage(self))

    def show(self):
        return self.model.parameters()

    def execute(self):
        print("Press any key to show the best")
        input()
        eval(self)
        return self.model.parameters()


def getImage(bird: Bird):
    bird.img_count += 1

    # For animation of bird, loop through three images
    if bird.img_count <= bird.ANIMATION_TIME:
        img = IMGS[0]
    elif bird.img_count <= bird.ANIMATION_TIME * 2:
        img = IMGS[1]
    elif bird.img_count <= bird.ANIMATION_TIME * 3:
        img = IMGS[2]
    elif bird.img_count <= bird.ANIMATION_TIME * 4:
        img = IMGS[1]
    elif bird.img_count == bird.ANIMATION_TIME * 4 + 1:
        img = IMGS[0]
        bird.img_count = 0

    # so when bird is nose diving it isn't flapping
    if bird.tilt <= -80:
        img = IMGS[1]
        bird.img_count = bird.ANIMATION_TIME * 2

    return img


class Pipe():
    """
    represents a pipe object
    """
    GAP = 200
    VEL = 5

    def __init__(self, x):
        """
        initialize pipe object
        :param x: int
        :param y: int
        :return" None
        """
        self.x = x
        self.height = 0

        # where the top and bottom of the pipe is
        self.top = 0
        self.bottom = 0

        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img

        self.passed = False

        self.set_height()

    def set_height(self):
        """
        set the height of the pipe, from the top of the screen
        :return: None
        """
        self.height = random.randrange(50, 450)
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP

    def move(self):
        """
        move pipe based on vel
        :return: None
        """
        self.x -= self.VEL

    def draw(self, win):
        """
        draw both the top and bottom of the pipe
        :param win: pygame window/surface
        :return: None
        """
        # draw top
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw bottom
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    def collide(self, bird, win):
        """
        returns if a point is colliding with the pipe
        :param bird: Bird object
        :return: Bool
        """
        bird_mask = bird.get_mask()
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)
        top_offset = (self.x - bird.x, self.top - round(bird.y))
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        b_point = bird_mask.overlap(bottom_mask, bottom_offset)
        t_point = bird_mask.overlap(top_mask, top_offset)

        if b_point or t_point:
            return True

        return False


class Base:
    """
    Represnts the moving floor of the game
    """
    VEL = 5
    WIDTH = base_img.get_width()
    IMG = base_img

    def __init__(self, y):
        """
        Initialize the object
        :param y: int
        :return: None
        """
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    def move(self):
        """
        move floor so it looks like its scrolling
        :return: None
        """
        self.x1 -= self.VEL
        self.x2 -= self.VEL
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    def draw(self, win):
        """
        Draw the floor. This is two images that move together.
        :param win: the pygame surface/window
        :return: None
        """
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))


def blitRotateCenter(surf, image, topleft, angle):
    """
    Rotate a surface and blit it to the window
    :param surf: the surface to blit to
    :param image: the image surface to rotate
    :param topLeft: the top left position of the image
    :param angle: a float value for angle
    :return: None
    """
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center=image.get_rect(topleft=topleft).center)

    surf.blit(rotated_image, new_rect.topleft)


def draw_window(win, birds, pipes, base, score, pipe_ind):
    """
    draws the windows for the best fitted at the end of the algorithm
    :param win: pygame window surface
    :param bird: a Bird object
    :param pipes: List of pipes
    :param score: score of the game (int)
    :param pipe_ind: index of closest pipe
    :return: None
    """
    win.blit(bg_img, (0, 0))

    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)
    for bird in birds:
        # draw lines from bird to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255, 0, 0),
                                 (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2),
                                 (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width() / 2, pipes[pipe_ind].height),
                                 5)
                pygame.draw.line(win, (255, 0, 0),
                                 (bird.x + bird.img.get_width() / 2, bird.y + bird.img.get_height() / 2), (
                                     pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width() / 2,
                                     pipes[pipe_ind].bottom), 5)
            except:
                pass
        # draw bird
        bird.draw(win)

    # score
    score_label = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    pygame.display.update()


class BirdEvaluator(SimpleIndividualEvaluator):
    def _evaluate_individual(self, individual: Bird):
        return eval(individual, show_game=False)


def eval(individual: Bird, limit=300, show_game=True):
    """
        Compute the fitness value of a given individual.
        Parameters
        Simulates a playable run
        ----------
        individual: Bird
            The individual to compute the fitness value for.
        Returns
        -------
        float
            The evaluated fitness value of the given individual, That is , increasing function of the distance reached.
    """
    global WIN, FLOOR
    win = WIN

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0
    curr_fitness = 0

    clock = pygame.time.Clock()

    run = True
    while run:
        clock.tick(30)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        pipe_ind = 0
        if len(pipes) > 1 and individual.x > pipes[0].x + pipes[0].PIPE_TOP.get_width():  # determine whether to
            # use the first or second
            pipe_ind = 1  # pipe on the screen for neural network input

        curr_fitness += 0.05
        individual.move()

        # send bird location, top pipe location and bottom pipe location and determine from network whether to
        # jump or not
        output = individual.model(
            (individual.y, abs(individual.y - pipes[pipe_ind].height), abs(individual.y - pipes[pipe_ind].bottom)))

        if output > 0:  # we used a tanh activation function so result will be between -1 and 1. if over 0 jump
            individual.jump()

        base.move()

        rem = []
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            # check for collision
            if pipe.collide(individual, win):
                curr_fitness -= 1
                #print(curr_fitness)
                return curr_fitness  # lost game but collided

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            if not pipe.passed and pipe.x < individual.x:
                pipe.passed = True
                add_pipe = True

        if add_pipe:  # passed pipe
            score += 1
            # can add this line to give more reward for passing through a pipe (not required)
            curr_fitness += 1.5  # give reward for passing pipe
            pipes.append(Pipe(WIN_WIDTH))

        for r in rem:
            pipes.remove(r)

        if individual.y + getImage(
                individual).get_height() - 10 >= FLOOR or individual.y < -50:  # indiviual escapes from screen
           # print(curr_fitness)
            return curr_fitness  # lost game

        if curr_fitness > limit:
            return curr_fitness
        print(curr_fitness)
        if show_game:
            draw_window(WIN, [individual], pipes, base, score, pipe_ind)


class BirdCreator(Creator):

    def __init__(self, init_pos: tuple, events=None):
        self.init_pos = init_pos
        if events is None:
            events = ["after_creation"]
        super().__init__(events)

    def create_individuals(self, n_individuals, higher_is_better):
        individuals = [Bird(x=self.init_pos[0],
                            y=self.init_pos[1],
                            model=FFModel().double(),
                            fitness=SimpleFitness(higher_is_better=higher_is_better))
                       for _ in range(n_individuals)]
        self.created_individuals = individuals

        return individuals


def main():
    algo = SimpleEvolution(
        Subpopulation(creators=BirdCreator(init_pos=(230, 350)),
                      population_size=80,
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
        max_workers=4,
        max_generation=5,

        termination_checker=ThresholdFromTargetTerminationChecker(optimal=300, threshold=0.0),
        statistics=BestAverageWorstStatistics()
    )

    # evolve the generated initial population
    algo.evolve()

    # Execute (show) the best solution
    print(algo.execute())


if __name__ == '__main__':
    main()
