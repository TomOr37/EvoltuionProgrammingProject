import pygame
import os
from eckity.evaluators.simple_individual_evaluator import SimpleIndividualEvaluator
from eckity.creators.creator import Creator
from eckity.fitness.simple_fitness import SimpleFitness
from feedForwardModel import FFModel
from eckity.individual import Individual
from gameObjects import Pipe , Base
from gameUtils import draw_window , blitRotateCenter

pygame.font.init()  # init font

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)


WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird" + str(x) + ".png"))) for x in
               range(1, 4)]

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
            return curr_fitness  # lost game

        if curr_fitness > limit:
            return curr_fitness

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


