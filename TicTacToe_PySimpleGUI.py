import PySimpleGUI as sg
from random import choice


def disable_buttons(board):
    for i in range(3):
        for j in range(3):
            board[(i, j)].Update(disabled=True)


def reset_board(board):
    for j in range(3):
        for i in range(3):
            board[(i, j)].Update('', disabled=False, button_color='ivory2')


def check_winner(board, table, turn):
    for j in range(3):
        if (0, j) in table.keys() and (1, j) in table.keys() and (2, j) in table.keys():
            if table[(0, j)] == table[(1, j)] == table[(2, j)]:
                [board[(k, j)].Update(button_color='olivedrab3') for k in range(3)]
                sg.Popup(f'VICTORY! The winner is {turn}. Please press Reset to play again or Quit to close the game window.')
                disable_buttons(board)

    for i in range(3):
        if (i, 0) in table.keys() and (i, 1) in table.keys() and (i, 2) in table.keys():
            if table[(i, 0)] == table[(i, 1)] == table[(i, 2)]:
                [board[(i, k)].Update(button_color='olivedrab3') for k in range(3)]
                sg.Popup(f'VICTORY! The winner is {turn}. Please press Reset to play again or Quit to close the game window.')
                disable_buttons(board)

    if (0, 0) in table.keys() and (1, 1) in table.keys() and (2, 2) in table.keys():
        if table[(0, 0)] == table[(1, 1)] == table[(2, 2)]:
            [board[(k, k)].Update(button_color='olivedrab3') for k in range(3)]
            sg.Popup(f'VICTORY! The winner is {turn}. Please press Reset to play again or Quit to close the game window.')
            disable_buttons(board)

    if (0, 2) in table.keys() and (1, 1) in table.keys() and (2, 0) in table.keys():
        if table[(0, 2)] == table[(1, 1)] == table[(2, 0)]:
            [board[(k, 2-k)].Update(button_color='olivedrab3') for k in range(3)]
            sg.Popup(f'VICTORY! The winner is {turn}. Please press Reset to play again or Quit to close the game window.')
            disable_buttons(board)


def check_tie(table):
    k = 0
    for i in range(3):
        for j in range(3):
            if (i, j) in table.keys():
                k += 1
    if k == 9:
        sg.Popup("IT'S A TIE! Please press Reset to play again or Quit to close the game window.")


def main_game():
    table = {}
    turn = choice(players)
    sym = 'O' if turn == players[0] else 'X'

    layout = [[sg.Button(key=(i, j), size=(8, 4), font=('Arial', 13), button_color='ivory2') for j in range(3)] for i in range(3)]
    layout.append([sg.Text(f'This turn belongs to {turn} using {sym}', key='player_update', expand_x=True)])
    layout.append([sg.Button('Reset', size=(9, 1)), sg.Button('Quit', size=(9, 1))])
    board = sg.Window('Tic-Tac-Toe', layout, finalize=True)
    sg.ChangeLookAndFeel('Black')
    board.set_cursor('X_cursor') if sym == 'X' else board.set_cursor('circle')

    while True:
        event, values = board.read()
        if event == sg.WIN_CLOSED or event == 'Quit':
            break
        elif event == 'Reset':
            reset_board(board)
            table = {}
        else:
            table[event] = sym
            board[event].Update(sym, disabled=True)
            check_winner(board, table, turn)
            check_tie(table)
            turn = players[1] if sym == 'O' else players[0]
            sym = 'O' if turn == players[0] else 'X'
            board.set_cursor('X_cursor') if sym == 'X' else board.set_cursor('circle')
            board['player_update'].Update(f'This turn belongs to {turn} using {sym}')

    board.close()


lay_init = [[sg.Text('Please set a name for both players.', font=('Arial', 11, 'bold'))],
            [sg.Text()],
            [sg.Text('Name player for "O": '), sg.InputText(size=(12, 1), expand_x=True, key='player_o')],
            [sg.Text()],
            [sg.Text('Name player for "X": '), sg.InputText(size=(12, 1), expand_x=True, key='player_x')],
            [sg.Text()],
            [sg.Button('Start game'), sg.Button('Cancel')]]
board_init = sg.Window('Name players', lay_init)
sg.ChangeLookAndFeel('Black')

while True:
    event_init, values_init = board_init.read()
    if event_init == sg.WIN_CLOSED or event_init == 'Cancel':
        break

    if event_init == 'Start game':
        if values_init['player_x'].strip() == '' or values_init['player_o'].strip() == '':
            sg.popup_error('Please input names for player(s)!')

        elif values_init['player_x'].strip() == values_init['player_o'].strip():
            sg.popup_error('Please name the players differently from each other!')

        else:
            players = [values_init['player_o'], values_init['player_x']]
            main_game()

board_init.close()
