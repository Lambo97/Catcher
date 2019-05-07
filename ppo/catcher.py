import numpy as np
from random import random


def percent_round_int(percent, x):
    return np.round(percent * x).astype(int)


class MyBar():

    """
        Wrap the parameters and the dynamics related
        to the bar the agent interacts with.
    """

    def __init__(self, width, height, grid_width, grid_height):
        """
        Set the parameters of the bar (size) and its initial position
        Arguments:
        ----------
        - `width`: Bar width
        - `height`: Bar height
        - `grid_width`: Grid width
        - `grid_height`: Grid height
        """
        self.width = width
        self.height = height

        self.grid_width = grid_width
        self.grid_height = grid_height
        self.vel = 0.0

        self.size = (width, height)
        self.center = (
            grid_width / 2 - self.width / 2,
            grid_height - height - 3)

    def reset(self):
        """
        Resets to initial state of the bar.
        Center position, null speed.
        """
        self.center = (
            self.grid_width / 2 - self.width / 2,
            self.grid_height - self.height - 3)
        self.vel = 0.0

    def update(self, dx):
        """
            Dynamics of the bar.
            Arguments:
            ----------
            - `dx`: Real-valued force
                    (towards left when <0, right otherwise)
        """

        # Update velocity and position
        self.vel += dx
        self.vel *= 0.9

        x, y = self.center
        n_x = x + self.vel

        """
            Keeps the bar inside the grid.
            No bounce, null speed if grid limits
            are reached.
        """
        if n_x <= 0:
            self.vel = 0.0
            n_x = 0

        if n_x + self.width >= self.grid_width:
            self.vel = 0.0
            n_x = self.grid_width - self.width

        self.center = (n_x, y)


class MyFruit():
    """
        Wrap the parameters and the dynamics related
        to the fruits.
    """

    def __init__(self, speed, size, grid_width, grid_height):
        """
        Set the parameters of the fruit (size, speed) and its initial position
        Arguments:
        ----------
        - `width`: Bar width
        - `height`: Bar height
        - `grid_width`: Grid width
        - `grid_height`: Grid height
        """

        self.speed = speed
        self.size = (size, size)

        self.grid_width = grid_width
        self.grid_height = grid_height

        """
        Defines ranges where the fruit can pop
        at the beginning of an episode and
        whenever it has been caught/miss by the bar
        """
        self.x_range = range(
            size *
            2,
            self.grid_width -
            size *
            2,
            size)
        self.y_range = range(
            size,
            int(self.grid_height / 2),
            size)

    def update(self, dt):
        """
        Updates the position of the fruit at a constant speed
        Arguments:
        ----------
        - `dt`: (single-step) integration constant
        """

        # Updates fruit position
        x, y = self.center
        n_y = y + self.speed * dt

        self.center = (x, n_y)

    def reset(self):
        """
            Resets to initial state, randomly somewhere in the grid,
            according to ranges defined by inner variables.
        """
        a, b = np.random.random((2,))
        x = np.floor(a * (self.grid_width - self.size[0]))
        y = np.floor(b * ((self.grid_height - self.size[1]) / 2.0))

        self.center = (x, -1 * y)


class ContinuousCatcher():
    """
        Based on `Eder Santana's` game idea.
        (https://github.com/EderSantana)
    """

    @staticmethod
    def gamma():
        """
            Returns discount factor
        """
        return 0.95

    def __init__(self, width=64, height=64, init_lives=3, dt=30):
        """
        Wrapper for the full dynamics/parameters set of the game
        Arguments:
        ----------
        - `width`: Grid width
        - `height`: Grid height
        - `init_lives`: Number of allowed missed fruits before game over
        - `dt`: Frame per second (used as integration constant)
        """
        self.width = width
        self.height = height
        self.dt = dt
        self.fps = self.dt
        self.dx = 0.0
        self.init_lives = init_lives

        # Parameters of the fruit
        self.fruit_size = percent_round_int(height, 0.06)
        self.fruit_fall_speed = 0.00095 * height

        # Parameters of the bar
        self.bar_speed = 0.021 * width
        self.bar_width = percent_round_int(width, 0.2)
        self.bar_height = percent_round_int(height, 0.04)

        # Reward function
        self.rtick = 1.0
        self.rpos = 2.0
        self.rneg = -2.0
        self.rloss = 0.0

        # Builds the bar with its parameters
        self.bar = MyBar(self.bar_width,
                         self.bar_height, self.width, self.height)
        self.bar_update = self.bar.update
        self.bar_reset = self.bar.reset

        # Builds the fruit with its parameters
        self.fruit = MyFruit(self.fruit_fall_speed, self.fruit_size,
                             self.width, self.height)
        self.fruit_update = self.fruit.update
        self.fruit_reset = self.fruit.reset

        self.viewer = None

    def reset(self):
        """
            Resets the game back to its initial state
        """
        self.lives = self.init_lives
        self.fruit_reset()
        self.bar_reset()
        return self.observe()

    def _collide_fruit(self):
        """
            Determines whether the bar hits the fruit
        """
        x1, y1 = self.bar.center
        x2, y2 = self.fruit.center
        w1, h1 = self.bar.size
        w2, h2 = self.fruit.size

        return (
            y1 < y2 +
            h2 and y1 +
            h1 > y2 and x1 < x2 +
            w2 and x1 +
            w1 > x2)

    def step(self, act):
        """
            Update the game with respect to its dynamics
        """
        done = False

        # Clip the absolute force to the maximum bar speed
        self.dx = np.clip(act[0], -self.bar_speed, self.bar_speed)

        # Grant reward related to tick and
        # whether fruit has been caught/missed
        reward = self.rtick
        if self.fruit.center[1] >= self.height:
            self.lives -= 1
            reward += self.rneg
            self.fruit_reset()

        if self._collide_fruit():
            self.fruit_reset()
            reward += self.rpos

        # Update bar and fruits
        self.bar_update(self.dx)
        self.fruit_update(self.fps)

        # Game over is reached when number of fruits have
        # trespassed a given thresold
        if self.lives == 0:
            reward += self.rloss
            done = True

        return self.observe(), reward, done

    def observe(self):
        """
            Returns the current game state
        """
        return np.asarray([self.bar.center[0], self.bar.vel,
                           self.fruit.center[0], self.fruit.center[1]])

    def render(self, score=0, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.width, self.height)
            self.viewer.set_bounds(0, self.width, 0, self.height)

            self.lb = self.bar.center[0]
            self.tb = self.height - self.bar.center[1]
            self.rb = self.lb + self.bar_width
            self.bb = self.tb + self.bar_height
            bar = rendering.make_polygon(
                [(self.lb, self.bb), (self.lb, self.tb), (self.rb, self.tb), (self.rb, self.bb)])
            bar.set_color(.8, .3, .3)
            self.bar_transform = rendering.Transform()
            bar.add_attr(self.bar_transform)
            self.viewer.add_geom(bar)

            self.lf = self.fruit.center[0]
            self.tf = self.height - self.fruit.center[1]
            self.rf = self.lf + self.fruit_size
            self.bf = self.tf + self.fruit_size
            fruit = rendering.make_polygon(
                [(self.lf, self.bf), (self.lf, self.tf), (self.rf, self.tf), (self.rf, self.bf)])
            fruit.set_color(1, 0.5, 0)
            self.fruit_transform = rendering.Transform()
            fruit.add_attr(self.fruit_transform)
            self.viewer.add_geom(fruit)

        self.bar_transform.set_translation(self.bar.center[0] - self.lb, 0)
        self.fruit_transform.set_translation(
            self.fruit.center[0] - self.lf,
            self.height - self.fruit.center[1] - self.tf)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    game = ContinuousCatcher(width=500, height=500)
    game.reset()
    print(game.observe())
    game.render()
    for i in range(200):
        game.render()
        action = random() * 6 - 3
        obs, reward, done = game.step([action])
    print(game.observe())
    game.close()
