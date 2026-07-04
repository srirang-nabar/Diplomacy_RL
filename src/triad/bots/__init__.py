"""Heuristic bots (CLAUDE.md §4.4 stage 0): pure policies for env opponents
and BC teachers. Every bot takes an explicit numpy Generator and breaks ties
uniformly at random — the engine is deterministic, so bot rng is the only
stochasticity in bot-vs-bot games."""
from triad.bots.base import Bot, play_game
from triad.bots.random_legal import RandomLegal
from triad.bots.grabber import Grabber
from triad.bots.turtle import Turtle

__all__ = ["Bot", "play_game", "RandomLegal", "Grabber", "Turtle"]
