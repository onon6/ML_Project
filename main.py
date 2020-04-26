import six
import pyspiel

game = pyspiel.load_game('matrix_mp')
state = game.new_initial_state()
print("State", state)