import PySimpleGUI as sg
from random import choice

ox = ['X', 'O']


def flip_player(sym):
    if sym == 'X':
        return 'O'
    if sym == 'O':
        return 'X'


def check_winner():
    c1, c2, c3 = result['pos1'], result['pos2'], result['pos3']
    c4, c5, c6 = result['pos4'], result['pos5'], result['pos6']
    c7, c8, c9 = result['pos7'], result['pos8'], result['pos9']
    if c1 == c2 == c3 or c4 == c5 == c6 or c7 == c8 == c9 or c1 == c4 == c7 or c2 == c5 == c8 or c3 == c6 == c9 or c1 == c5 == c9 or c3 == c5 == c7:
        if sg.popup_yes_no(f'Player "{sym}" has won! Do you wish to play again?') == 'Yes':
            return True
        else:
            return False


def check_draw():
    c1, c2, c3 = result['pos1'], result['pos2'], result['pos3']
    c4, c5, c6 = result['pos4'], result['pos5'], result['pos6']
    c7, c8, c9 = result['pos7'], result['pos8'], result['pos9']
    if c1 in ox and c2 in ox and c3 in ox and c4 in ox and c5 in ox and c6 in ox and c7 in ox and c8 in ox and c9 in ox:
        if sg.Popup('It is a draw!') == 'OK':
            return True


sym = choice(ox)
result = {'pos1': 1, 'pos2': 2, 'pos3': 3, 'pos4': 4, 'pos5': 5, 'pos6': 6, 'pos7': 7, 'pos8': 8, 'pos9': 9}

layout = [[sg.Button(size=(8, 4), key='pos1'), sg.Button(size=(8, 4), key='pos2'), sg.Button(size=(8, 4), key='pos3')],
          [sg.Button(size=(8, 4), key='pos4'), sg.Button(size=(8, 4), key='pos5'), sg.Button(size=(8, 4), key='pos6')],
          [sg.Button(size=(8, 4), key='pos7'), sg.Button(size=(8, 4), key='pos8'), sg.Button(size=(8, 4), key='pos9')]]

board = sg.Window('Tic-Tac-Toe', layout)
sg.ChangeLookAndFeel('Default1')

while True:
    event, values = board.read()
    if event == sg.WIN_CLOSED:
        break

    sym = flip_player(sym)
    board[event].Update(sym, disabled=True)
    result[event] = sym

    if check_winner() is False:
        break

    if check_draw() is True:
        break

board.close()
