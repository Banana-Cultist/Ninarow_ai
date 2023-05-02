from __future__ import annotations
from typing import *
import random

import pygame
import pygame.gfxdraw

WHITE = 1
BLACK = 0
EMPTY = -1

class Game:
    def __init__(self, width: int) -> None:
        self.width: Final = width
        self.board: list[list[int]] = [
            [
                EMPTY for _ in range(width)
            ]
            for _ in range(width)
        ]
    
    def randomize(self) -> None:
        for row in range(self.width):
            for col in range(self.width):
                self.board[row][col] = random.choice((WHITE, BLACK, EMPTY))
                
    def draw(self, screen: pygame.Surface) -> None:
        screen.fill((166, 103, 45))
        assert screen.get_width() == screen.get_height()
        line_color: Final = (36, 28, 21)
        screen_width: Final = screen.get_width()
        # margin_width: Final = screen_width*.06
        line_spacing: Final = (screen_width) / (self.width+1)
        piece_width: Final = line_spacing * .4
        for i in range(1, self.width+1):
            pygame.draw.line(
                screen,
                line_color,
                (
                    i*line_spacing,
                    line_spacing
                ),
                (
                    i*line_spacing,
                    screen_width - line_spacing
                ),
            )
            pygame.draw.line(
                screen,
                line_color,
                (
                    line_spacing,
                    i*line_spacing
                ),
                (
                    screen_width - line_spacing,
                    i*line_spacing
                ),
            )
        
        for row, line in enumerate(self.board):
            for col, piece in enumerate(line):
                if piece == EMPTY:
                    continue
                pygame.draw.circle(
                    screen,
                    (250, 250, 250) if piece == WHITE else (0, 0, 0),
                    (
                        screen_width * (col+1) / (self.width+1),
                        screen_width *  (row+1) / (self.width+1)
                    ),
                    piece_width,
                )
        
    
def main() -> None:
    game = Game(7)
    game.randomize()
    screen = pygame.display.set_mode((700, 700))
    
    while not pygame.QUIT in [event.type for event in pygame.event.get()]:
        game.draw(screen)
        pygame.display.flip()

if __name__ == '__main__':
    main()